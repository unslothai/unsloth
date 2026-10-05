# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused owned-process lifecycle tests for Clef."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from core.systemone import clef_runtime, clef_worker
from core.systemone.catalog import ClefCheckpoint
from core.systemone.laya_runtime import Unavailable


_REAL_CHECKPOINT_DIR = clef_runtime._checkpoint_dir


def _error(queue, phase, kind, message):
    queue.put({"type": "error", "phase": phase, "kind": kind, "message": message})


def _fake_worker(*, cmd_queue, resp_queue, cancel_event, ready_event=None, config=None):
    while True:
        command = cmd_queue.get()
        phase, model = command.get("type"), command.get("model")
        if phase == "shutdown":
            resp_queue.put({"type": "shutdown_ack"})
            return
        if phase == "load":
            if model == "clef-fail":
                _error(resp_queue, phase, "worker_error", "planned load failure")
                return
            resp_queue.put({"type": "loaded", "device": "cpu", "gpu_available": False})
            continue
        if model == "clef-hang":
            cancel_event.wait()
            _error(resp_queue, phase, "cancelled", "cancelled")
            continue
        result = {"model": model, "answers": {}, "usage": {"input_tokens": 7, "output_tokens": 0}}
        result["echo"] = {"state": command["state"], "questions": command["questions"]}
        resp_queue.put({"type": "result", "result": result})


def _checkpoint(name="clef-test"):
    return ClefCheckpoint(name, f"Cloudflare/{name}", None, "test", 123, "a" * 40, ("config.json",))


def _wait(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.02)
    assert predicate(), "condition did not become true"


class _Registry:
    def __init__(self, granted=True, state="owned"):
        self.granted, self.state, self.events, self.releases, self.owner, self.process = (
            granted,
            state,
            [],
            [],
            None,
            None,
        )

    def claim_repository_owner(self, repo, owner):
        self.events.append("claim")
        if self.granted:
            self.owner = owner
        return self.granted, self.state

    def release_repository_owner(self, repo, owner):
        assert self.process is None or not self.process.is_alive()
        self.releases.append((repo, owner))
        return owner is self.owner


@pytest.fixture(autouse=True)
def _runtime(monkeypatch, tmp_path):
    clef_runtime.shutdown()
    with clef_runtime._state_lock:
        clef_runtime._shutdown_requested = False
        for name in ("_failure", "_loading", "_worker", "_lease", "_loaded", "_device_name"):
            setattr(clef_runtime, name, None)
        clef_runtime._generation += 1
    monkeypatch.setattr(clef_runtime, "_admission", threading.BoundedSemaphore(8))
    monkeypatch.setattr(
        clef_runtime, "_WORKER_FACTORY", lambda: clef_runtime.ClefWorker(_fake_worker)
    )
    monkeypatch.setattr(clef_runtime, "_checkpoint_dir", lambda _checkpoint: tmp_path)
    yield
    clef_runtime.shutdown()
    with clef_runtime._state_lock:
        clef_runtime._shutdown_requested = False


@pytest.mark.parametrize("idle", [False, True])
def test_worker_holds_cache_lease_until_unload_or_idle_retirement(monkeypatch, tmp_path, idle):
    import hub.utils.download_registry as downloads

    registry = _Registry()
    monkeypatch.setattr(downloads, "get_models_registry", lambda: registry)
    monkeypatch.setattr(
        clef_runtime, "_checkpoint_dir", lambda _: registry.events.append("resolve") or tmp_path
    )
    if idle:
        monkeypatch.setattr(clef_runtime, "IDLE_UNLOAD_S", 0.05)
    checkpoint, state, questions = (
        _checkpoint(),
        {"text": "café", "none": None},
        {"q": {"type": "noul"}},
    )
    clef_runtime.prepare(checkpoint)
    assert clef_runtime.decide(checkpoint, state, questions, [])["echo"] == {
        "state": state,
        "questions": questions,
    }
    process, registry.process = clef_runtime._worker._process, clef_runtime._worker._process
    assert registry.events[:2] == ["claim", "resolve"] and not registry.releases
    if idle:
        _wait(lambda: clef_runtime.status()["loaded_model"] is None)
    else:
        assert clef_runtime.unload()
    _wait(lambda: bool(registry.releases))
    assert not process.is_alive() and registry.releases == [(checkpoint.source, registry.owner)]


def test_cache_claim_rejection_prevents_snapshot_resolution(monkeypatch):
    import hub.utils.download_registry as downloads

    registry, resolved = _Registry(False, "deleting"), []
    monkeypatch.setattr(downloads, "get_models_registry", lambda: registry)
    monkeypatch.setattr(clef_runtime, "_checkpoint_dir", lambda _: resolved.append(True))
    with pytest.raises(Unavailable, match="being deleted"):
        clef_runtime.decide(_checkpoint(), "state", {"q": {"type": "noul"}}, [])
    assert registry.events == ["claim"] and not resolved


def test_load_failure_and_pre_spawn_cancellation_leave_no_worker(monkeypatch, tmp_path):
    with pytest.raises(Unavailable, match="planned load failure"):
        clef_runtime.decide(_checkpoint("clef-fail"), "state", {"q": {"type": "noul"}}, [])
    entered, release, factories = threading.Event(), threading.Event(), []

    def blocked(_):
        entered.set()
        assert release.wait(5)
        return tmp_path

    monkeypatch.setattr(clef_runtime, "_checkpoint_dir", blocked)
    monkeypatch.setattr(
        clef_runtime,
        "_WORKER_FACTORY",
        lambda: factories.append(True) or clef_runtime.ClefWorker(_fake_worker),
    )
    clef_runtime.prepare(_checkpoint("clef-race"))
    assert entered.wait(5)
    thread = threading.Thread(target=clef_runtime.unload)
    thread.start()
    _wait(lambda: clef_runtime._loading is None)
    release.set()
    thread.join(5)
    assert not thread.is_alive() and not factories and clef_runtime.status()["loaded_model"] is None


def test_shutdown_force_retires_an_active_decision_worker():
    checkpoint, outcome = _checkpoint("clef-hang"), []

    def run():
        try:
            clef_runtime.decide(checkpoint, "state", {"q": {"type": "noul"}}, [])
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    _wait(lambda: clef_runtime.status()["loaded_model"] == checkpoint.name)
    process = clef_runtime._worker._process
    _wait(clef_runtime._run_lock.locked)
    with pytest.raises(Unavailable, match="Wait for the Clef decision"):
        clef_runtime.ensure_can_unload()
    clef_runtime.shutdown()
    thread.join(5)
    assert not thread.is_alive() and outcome and isinstance(outcome[0], Unavailable)
    _wait(lambda: not process.is_alive())


def test_local_cache_is_pinned_and_over_context_is_refused(monkeypatch, tmp_path):
    import huggingface_hub

    checkpoint, seen = _checkpoint(), {}
    for name in checkpoint.files:
        (tmp_path / name).write_text("x", encoding="utf-8")
    monkeypatch.setattr(clef_runtime, "_checkpoint_dir", _REAL_CHECKPOINT_DIR)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, **kwargs: seen.update(repo=repo, **kwargs) or str(tmp_path),
    )
    assert clef_runtime.is_cached(checkpoint)
    assert seen["revision"] == checkpoint.revision and seen["local_files_only"] is True

    class Reference:
        @staticmethod
        def encode_record(*_args, **_kwargs):
            return SimpleNamespace(input_ids=tuple(range(clef_worker.MAX_CONTEXT_TOKENS + 1)))

    with pytest.raises(ValueError, match="require 16385 tokens"):
        clef_worker.encode_record_untruncated(Reference, "tokenizer", {"state": "x"}, "processor")


def test_another_load_waits_until_the_retiring_process_exits(monkeypatch):
    clef_runtime.decide(_checkpoint(), "state", {"q": {"type": "noul"}}, [])
    worker = clef_runtime._worker
    close = worker.close
    entered, release = threading.Event(), threading.Event()

    def blocked_close(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return close(*args, **kwargs)

    monkeypatch.setattr(worker, "close", blocked_close)
    thread = threading.Thread(target=clef_runtime.unload)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(Unavailable, match="still stopping"):
            clef_runtime.prepare(_checkpoint("clef-next"))
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not worker.is_alive()
