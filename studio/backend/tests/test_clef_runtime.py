# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading
import time
from types import SimpleNamespace

import pytest

from core.systemone import clef_worker, owned_runtime as rt
from core.systemone.catalog import ClefCheckpoint
from core.systemone.laya_runtime import Unavailable

_CACHE_DIR = rt._checkpoint_dir
_QUESTIONS = {"q": {"type": "noul"}}


def _fake_worker(*, cmd_queue, resp_queue, cancel_event):
    from utils.process_lifetime import bind_current_process_to_parent_lifetime
    bind_current_process_to_parent_lifetime()
    while True:
        command = cmd_queue.get()
        phase, model = command.get("type"), command.get("model")
        if phase == "shutdown":
            resp_queue.put({"type": "shutdown_ack"})
            return
        if phase == "load" and model != "clef-fail":
            resp_queue.put({"type": "loaded", "device": "cpu", "gpu_available": False})
            continue
        if phase == "load" or model == "clef-hang":
            if phase != "load":
                cancel_event.wait()
            resp_queue.put({"type": "error", "phase": phase,
                            "kind": "worker_error" if phase == "load" else "cancelled",
                            "message": "planned load failure" if phase == "load" else "cancelled"})
            if phase == "load":
                return
            continue
        result = {"model": model, "answers": {}, "usage": {"input_tokens": 7, "output_tokens": 0},
                  "echo": {"state": command["state"], "questions": command["questions"]}}
        resp_queue.put({"type": "result", "result": result})


def _checkpoint(name = "clef-test"):
    return ClefCheckpoint(name, f"Cloudflare/{name}", None, "test", 123,
                          revision = "a" * 40, files = ("config.json",))


def _decide(name = "clef-test", state = "state", questions = _QUESTIONS):
    return rt.decide(_checkpoint(name), state, questions, [])


def _wait(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.02)
    assert predicate(), "condition did not become true"


class _Registry:
    def __init__(self, granted = True, state = "owned"):
        self.granted, self.state = granted, state
        self.events, self.releases = [], []
        self.owner = self.process = None

    def claim_repository_owner(self, repo, owner):
        self.events.append("claim")
        if self.granted:
            self.owner = owner
        return self.granted, self.state

    def release_repository_owner(self, repo, owner):
        assert self.process is None or not self.process.is_alive()
        self.releases.append((repo, owner))
        return owner is self.owner


@pytest.fixture(autouse = True)
def _runtime(monkeypatch, tmp_path):
    rt.shutdown()
    rt._shutdown_requested = False
    monkeypatch.setattr(rt, "_admission", threading.BoundedSemaphore(8))
    monkeypatch.setattr(rt, "_WORKER_FACTORY", lambda: rt.ClefWorker(_fake_worker))
    monkeypatch.setattr(rt, "_checkpoint_dir", lambda _: tmp_path)
    yield
    rt.shutdown()
    rt._shutdown_requested = False


@pytest.mark.parametrize("idle", [False, True])
def test_worker_holds_cache_lease_until_unload_or_idle_retirement(monkeypatch, tmp_path, idle):
    import hub.utils.download_registry as downloads
    registry = _Registry()
    monkeypatch.setattr(downloads, "get_models_registry", lambda: registry)
    monkeypatch.setattr(rt, "_checkpoint_dir", lambda _: registry.events.append("resolve") or tmp_path)
    if idle:
        monkeypatch.setattr(rt, "IDLE_UNLOAD_S", 0.05)
    rt.prepare(_checkpoint())
    if not idle:
        thread = rt._loading.thread
        thread.join(5)
        assert not thread.is_alive()
        time.sleep(0.1)
        assert rt._worker.is_alive(), "The worker must outlive the loader thread"
    state = {"text": "café", "none": None}
    assert _decide(state = state)["echo"] == {"state": state, "questions": _QUESTIONS}
    process = registry.process = rt._worker._process
    assert registry.events[:2] == ["claim", "resolve"] and not registry.releases
    if idle:
        _wait(lambda: rt.status()["loaded_model"] is None)
    else:
        assert rt.unload()
    _wait(lambda: bool(registry.releases))
    assert not process.is_alive() and registry.releases == [(_checkpoint().source, registry.owner)]


def test_cache_claim_rejection_prevents_snapshot_resolution(monkeypatch):
    import hub.utils.download_registry as downloads
    registry, resolved = _Registry(False, "deleting"), []
    monkeypatch.setattr(downloads, "get_models_registry", lambda: registry)
    monkeypatch.setattr(rt, "_checkpoint_dir", lambda _: resolved.append(True))
    with pytest.raises(Unavailable, match = "being deleted"):
        _decide()
    assert registry.events == ["claim"] and not resolved


def test_load_failure_and_pre_spawn_cancellation_leave_no_worker(monkeypatch, tmp_path):
    with pytest.raises(Unavailable, match = "planned load failure"):
        _decide("clef-fail")
    entered, release, factories = threading.Event(), threading.Event(), []
    def blocked(_):
        entered.set()
        assert release.wait(5)
        return tmp_path
    monkeypatch.setattr(rt, "_checkpoint_dir", blocked)
    monkeypatch.setattr(rt, "_WORKER_FACTORY", lambda: factories.append(True) or rt.ClefWorker(_fake_worker))
    rt.prepare(_checkpoint("clef-race"))
    assert entered.wait(5)
    thread = threading.Thread(target = rt.unload)
    thread.start()
    try:
        _wait(lambda: rt._loading is None)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not factories and rt.status()["loaded_model"] is None


def test_shutdown_force_retires_an_active_decision_worker():
    outcome = []
    def run():
        try:
            _decide("clef-hang")
        except BaseException as exc:
            outcome.append(exc)
    thread = threading.Thread(target = run)
    thread.start()
    _wait(lambda: rt.status()["loaded_model"] == "clef-hang")
    process = rt._worker._process
    _wait(rt._run_lock.locked)
    with pytest.raises(Unavailable, match = "Wait for the Clef decision"):
        rt.ensure_can_unload()
    rt.shutdown()
    thread.join(5)
    assert not thread.is_alive() and outcome and isinstance(outcome[0], Unavailable)
    _wait(lambda: not process.is_alive())


def test_local_cache_is_pinned_and_over_context_is_refused(monkeypatch, tmp_path):
    import huggingface_hub
    checkpoint, seen = _checkpoint(), {}
    for name in checkpoint.files:
        (tmp_path / name).write_text("x", encoding = "utf-8")
    monkeypatch.setattr(rt, "_checkpoint_dir", _CACHE_DIR)
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **kwargs: seen.update(repo = repo, **kwargs) or str(tmp_path))
    assert rt.is_cached(checkpoint)
    assert seen["revision"] == checkpoint.revision and seen["local_files_only"] is True
    reference = SimpleNamespace(encode_record = lambda *a, **kw: SimpleNamespace(
        input_ids = range(clef_worker.MAX_CONTEXT_TOKENS + 1)))
    with pytest.raises(ValueError, match = "require 16385 tokens"):
        clef_worker.encode_record_untruncated(reference, "tokenizer", {"state": "x"}, "processor")


def test_another_load_waits_until_the_retiring_process_exits(monkeypatch):
    _decide()
    worker, entered, release = rt._worker, threading.Event(), threading.Event()
    close = worker.close
    def blocked_close(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return close(*args, **kwargs)
    monkeypatch.setattr(worker, "close", blocked_close)
    thread = threading.Thread(target = rt.unload)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(Unavailable, match = "still stopping"):
            rt.prepare(_checkpoint("clef-next"))
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not worker.is_alive()


@pytest.mark.parametrize("retirement", ["unload", "idle", "handoff"])
def test_gpu_retirement_does_not_evict_a_later_cpu_worker(monkeypatch, retirement):
    from core.inference import gpu_arbiter as arb
    for name, value in (("_owner", None), ("_owner_account", None),
                        ("other_accounts_active", lambda _: 0), ("_release_idle_video_servers", lambda _: None)):
        monkeypatch.setattr(arb, name, value)
    _decide()
    monkeypatch.setattr(rt, "_device_name", "cuda")
    arb.acquire_for(arb.DECISIONS)
    worker = rt._worker
    if retirement == "handoff":
        thread = threading.Thread(target = lambda: arb.acquire_for(arb.CHAT))
        thread.start()
        thread.join(5)
        assert not thread.is_alive(), "Eviction must not re-enter the arbiter lock"
    elif retirement == "idle":
        rt._idle_unload(worker, rt._generation)
    else:
        rt.unload()
    _wait(lambda: arb.current_owner() == (arb.CHAT if retirement == "handoff" else None))
    assert not worker.is_alive()
    _decide("clef-cpu")
    cpu_worker = rt._worker
    arb.acquire_for(arb.CHAT, allow_evict = False)
    assert cpu_worker.is_alive(), "A GPU acquisition must leave CPU Clef resident"


@pytest.mark.parametrize("accelerator", [None, "cuda", "xpu", "mps"])
@pytest.mark.parametrize("requested", ["cpu", "gpu"])
def test_pytorch_device_selection_never_silently_falls_back(accelerator, requested):
    def api(name):
        return SimpleNamespace(is_available = lambda: accelerator == name, is_bf16_supported = lambda: True)
    torch = SimpleNamespace(cuda = api("cuda"), xpu = api("xpu"), backends = SimpleNamespace(mps = api("mps")),
                            float32 = "fp32", float16 = "fp16", bfloat16 = "bf16")
    if requested == "gpu" and accelerator is None:
        with pytest.raises(clef_worker.ClefWorkerError, match = "GPU"):
            clef_worker._actual_device(torch, requested)
    else:
        device, dtype, available = clef_worker._actual_device(torch, requested)
        assert available is (accelerator is not None)
        assert (device, dtype) == (("cpu", "fp32") if requested == "cpu" else
                                   ("mps", "fp16") if accelerator == "mps" else (f"{accelerator}:0", "bf16"))


def test_shutdown_racing_adoption_closes_the_worker(monkeypatch, tmp_path):
    from utils import process_lifetime
    adopted, adopt = [], process_lifetime.adopt_pid
    monkeypatch.setattr(process_lifetime, "adopt_pid", lambda pid: adopted.append(pid) or adopt(pid))
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", lambda: bool(adopted))
    worker = rt.ClefWorker(_fake_worker)
    with pytest.raises(rt.ClefWorkerCancelled, match = "cancelled"):
        worker.start(tmp_path, _checkpoint(), "cpu", threading.Event())
    assert adopted and not worker.is_alive()
