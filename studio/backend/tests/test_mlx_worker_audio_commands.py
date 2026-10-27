# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A dropped audio command costs the caller its full 120s deadline, so the MLX loop must always reply."""

import os
import queue as _queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import worker


class _CmdQueue:
    """Feeds a fixed script, then behaves like an idle mp.Queue."""

    def __init__(self, cmds):
        self._cmds = list(cmds)

    def get(self, timeout = None):
        if self._cmds:
            return self._cmds.pop(0)
        raise _queue.Empty


class _RespQueue:
    def __init__(self):
        self.sent = []

    def put(self, item, *a, **k):
        self.sent.append(item)


def _run_mlx_loop(
    monkeypatch,
    cmds,
    *,
    backend = None,
    command_queue = None,
    **kwargs,
):
    """Drive the real MLX command loop with the init short-circuited."""
    from utils.hardware import hardware as _hw

    monkeypatch.setenv("ENVIRONMENT_TYPE", "development")
    monkeypatch.setattr(worker, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(worker, "apply_gpu_ids", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_recorded_local_base", lambda m: (None, False))
    monkeypatch.setattr(worker, "_hub_targets_are_local", lambda *a, **k: True)
    monkeypatch.setattr(worker, "_activate_transformers_version", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_handle_load", lambda *a, **k: None)
    monkeypatch.setattr(_hw, "detect_hardware", lambda *a, **k: None)
    monkeypatch.setattr(_hw, "DEVICE", _hw.DeviceType.MLX)

    import core.inference.mlx_inference as mlx_mod

    monkeypatch.setattr(
        mlx_mod,
        "MLXInferenceBackend",
        lambda *a, **k: backend if backend is not None else SimpleNamespace(),
    )

    from loggers.config import LogConfig

    monkeypatch.setattr(LogConfig, "setup_logging", staticmethod(lambda *a, **k: None))

    resp = _RespQueue()
    worker.run_inference_process(
        cmd_queue = command_queue or _CmdQueue([*cmds, {"type": "shutdown"}]),
        resp_queue = resp,
        cancel_event = SimpleNamespace(is_set = lambda: False, clear = lambda: None, set = lambda: None),
        config = {"model_name": "unsloth/orpheus-3b-0.1-ft"},
        **kwargs,
    )
    return resp.sent


def test_mlx_loop_refuses_a_tts_command_instead_of_dropping_it(monkeypatch):
    """`generate_audio` has no MLX handler. Falling through leaves the parent
    blocked for the full 120s deadline with nothing to report."""
    sent = _run_mlx_loop(monkeypatch, [{"type": "generate_audio", "request_id": "r1"}])

    errors = [m for m in sent if m.get("type") in ("audio_error", "error")]
    assert errors, f"the TTS command produced no reply at all: {sent}"
    assert (
        errors[0]["request_id"] == "r1"
    ), "the reply must carry the request_id or the direct-reader mailbox drops it"
    assert "MLX" in errors[0]["error"]


def test_mlx_loop_reports_an_unknown_command(monkeypatch):
    """Terminal branch, matching the GPU loop: never drop a command silently."""
    sent = _run_mlx_loop(monkeypatch, [{"type": "generate_video", "request_id": "r2"}])

    errors = [m for m in sent if m.get("type") == "error"]
    assert errors, f"the unknown command produced no reply at all: {sent}"
    assert errors[0]["request_id"] == "r2"
    assert "generate_video" in errors[0]["error"]


def test_whisper_on_a_backend_without_asr_explains_itself(monkeypatch):
    """The bare AttributeError names an internal method; the user needs the reason."""
    backend = SimpleNamespace()
    resp = _RespQueue()

    worker._handle_generate_audio_input(
        backend,
        {"request_id": "r3", "audio_data": [0.0, 0.0], "audio_type": "whisper"},
        resp,
        SimpleNamespace(is_set = lambda: False),
    )

    errors = [m for m in resp.sent if m.get("type") == "gen_error"]
    assert errors, resp.sent
    assert "not supported on the MLX backend" in errors[0]["error"]
    assert "attribute" not in errors[0]["error"].lower()


def test_mlx_loop_answers_every_prompt_cache_resume_command(monkeypatch):
    backend = SimpleNamespace(
        save_prompt_cache_for_resume = lambda directory: {"dir": directory},
        restore_prompt_cache_for_resume = lambda manifest: (_ for _ in ()).throw(
            RuntimeError("unreadable")
        ),
    )
    sent = _run_mlx_loop(
        monkeypatch,
        [
            {"type": "save_prompt_cache", "request_id": "s", "directory": "/d"},
            {"type": "restore_prompt_cache", "request_id": "r", "manifest": {}},
        ],
        backend = backend,
    )
    replies = {m["request_id"]: m for m in sent if m.get("request_id") in ("s", "r")}
    assert replies["s"]["type"] == "save_prompt_cache_response"
    assert replies["s"]["result"] == {"dir": "/d"}
    assert replies["r"]["type"] == "restore_prompt_cache_response"
    assert replies["r"]["error"] == "unreadable"

    resp = _RespQueue()
    worker._handle_prompt_cache_resume(
        SimpleNamespace(), {"type": "save_prompt_cache", "request_id": "u"}, resp
    )
    assert resp.sent[0]["request_id"] == "u" and "no prompt cache" in resp.sent[0]["error"]


@pytest.mark.parametrize("case", ["saved", "error", "abandoned", "dispatched"])
def test_a_prompt_cache_save_keeps_its_directory_only_with_a_manifest(monkeypatch, tmp_path, case):
    import threading

    from core.inference import orchestrator as orch_mod
    from utils.paths import storage_roots

    root = tmp_path / "mlx-prompt-cache"
    monkeypatch.setattr(storage_roots, "mlx_prompt_cache_root", lambda: root)
    orch = orch_mod.InferenceOrchestrator.__new__(orch_mod.InferenceOrchestrator)
    orch.active_model_name, orch.models = "m", {"m": {"is_mlx": True}}
    orch._managed_engine, orch._unload_pending = None, False
    orch._gen_lock, orch._mailbox_lock = threading.Lock(), threading.Lock()
    orch._mailboxes = {"generation": object()} if case == "dispatched" else {}
    sent = []

    def reply(
        cmd,
        timeout,
        cancelled = None,
    ):
        sent.append(cmd)
        if case == "saved":
            return {"result": {"engine": "mlx", "dir": cmd["directory"]}}
        return {"error": "disk full"} if case == "error" else None

    monkeypatch.setattr(orch, "_prompt_cache_resume_reply", reply)
    manifest = orch.save_prompt_cache_for_resume()
    assert not orch._gen_lock.locked() and len(sent) == (case != "dispatched")
    if case == "saved":
        assert [str(path) for path in root.iterdir()] == [manifest["dir"]]
        if os.name == "posix":
            # Saved KV encodes chat content, so neither level is readable by other local users.
            assert {
                (root.stat().st_mode & 0o777),
                (Path(manifest["dir"]).stat().st_mode & 0o777),
            } == {0o700}
    else:
        assert manifest is None and list(root.iterdir() if root.exists() else ()) == []


@pytest.mark.parametrize("cmd_type", ["generate_audio", "generate_video"])
def test_every_mlx_command_gets_exactly_one_reply(monkeypatch, cmd_type):
    """One reply, not zero and not a duplicate that would confuse the mailbox."""
    sent = _run_mlx_loop(monkeypatch, [{"type": cmd_type, "request_id": "r4"}])

    addressed = [m for m in sent if m.get("request_id") == "r4"]
    assert len(addressed) == 1, addressed


@pytest.mark.parametrize(
    "setting, enabled",
    [(None, True), ("0", False), ("false", False), ("1", True), ("TRUE", True), ("yes", True)],
)
def test_mlx_idle_warmth_is_default_on_bounded_and_failure_safe(monkeypatch, setting, enabled):
    import sys
    from types import ModuleType

    now, ticks = [100.0], []
    monkeypatch.setattr(worker.time, "monotonic", lambda: now[0])
    if setting is None:
        monkeypatch.delenv("UNSLOTH_MLX_GPU_KEEP_WARM", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_MLX_GPU_KEEP_WARM", setting)
    core = ModuleType("mlx.core")
    core.float32 = object()
    core.zeros = lambda *a, **k: 0
    core.eval = lambda x: ticks.append(x)
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    warmth = worker._MLXIdleWarmth()
    warmth.tick(True)
    assert ticks == []
    warmth.active()
    assert warmth.timeout(True) == (0.5 if enabled else 1.0)
    warmth.tick(False)
    assert ticks == []
    warmth.tick(True)
    assert ticks == ([1] if enabled else [])
    ticks.clear()
    now[0] += 60
    warmth.tick(True)
    assert ticks == []
    warmth.active()
    warmth.clear()
    warmth.tick(True)
    assert ticks == []
    warmth.active()
    core.eval = lambda x: (_ for _ in ()).throw(RuntimeError("tick failure"))
    warmth.tick(True)
    assert not warmth.enabled
    assert warmth.timeout(True) == 1.0


@pytest.mark.parametrize("ending", ["unload", "reset", "cancel"])
@pytest.mark.parametrize("blocked", [None, "drain", "teardown", "busy"])
def test_mlx_loop_ticks_only_when_loaded_recent_and_idle(monkeypatch, blocked, ending):
    now, ticks = [100.0], []
    monkeypatch.setattr(worker.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("UNSLOTH_MLX_GPU_KEEP_WARM", "1")
    monkeypatch.setattr(
        worker._MLXIdleWarmth,
        "tick",
        lambda self, loaded: ticks.append(loaded) if self.timeout(loaded) == 0.5 else None,
    )
    backend = SimpleNamespace(active_model_name = "model", reset_generation_state = lambda: None)
    monkeypatch.setattr(worker, "_dispatch_generate", lambda *a, **k: None)
    monkeypatch.setattr(
        worker, "_handle_unload", lambda *a, **k: setattr(backend, "active_model_name", None)
    )
    blocked_now = [False]
    busy = [False]
    original_batch = worker._ResidentBatch

    class Batch(original_batch):
        @property
        def rows_in_flight(self):
            return int(busy[0])

        def step(self, waiting = None):
            pass

    monkeypatch.setattr(worker, "_ResidentBatch", Batch)

    class Commands:
        step = 0

        def get(self, timeout = None):
            self.step += 1
            if self.step == 1:
                return {"type": "generate", "request_id": "warm"}
            if self.step in (2, 3):
                now[0] += 0.5
                blocked_now[0] = self.step == 2 and blocked in ("drain", "teardown")
                busy[0] = self.step == 2 and blocked == "busy"
                raise _queue.Empty
            if self.step == 4:
                return {"type": ending}
            if self.step == 5:
                raise _queue.Empty
            return {"type": "shutdown"}

    _run_mlx_loop(
        monkeypatch,
        [],
        backend = backend,
        command_queue = Commands(),
        drain_event = SimpleNamespace(is_set = lambda: blocked == "drain" and blocked_now[0]),
        pending_teardowns = SimpleNamespace(
            any_in_flight = lambda: blocked == "teardown" and blocked_now[0], taken = lambda: None
        ),
    )
    assert ticks == (["model", "model"] if blocked is None else ["model"])
