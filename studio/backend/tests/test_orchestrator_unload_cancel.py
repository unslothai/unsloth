# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrator unload cancels an in-flight generation instead of waiting it out.

A model switch during a long generation used to queue the ``unload`` command
behind the running ``generate`` (the subprocess is sequential), hanging the UI
for the full generation. ``unload_model`` now cancels the generation first (the
mp.Event the worker checks each token) and takes ``_gen_lock`` before the unload
round-trip, so the switch is near-instant -- matching the GGUF backend.
"""
import threading
import time

import pytest

from core.inference import orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator


def _bare_orchestrator():
    """An orchestrator instance without the real __init__ subprocess/network."""
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._gen_lock = threading.Lock()
    o._cancel_event = threading.Event()  # stands in for the mp.Event (set/is_set/clear)
    o._proc = object()  # truthy so _ensure_subprocess_alive can report alive
    o._cmd_queue = object()
    o._resp_queue = object()
    o.active_model_name = "m"
    o.models = {"m": {}}
    o.loading_models = set()
    return o


def test_unload_cancels_inflight_generation_then_unloads(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: sent.append(cmd))
    monkeypatch.setattr(o, "_wait_response", lambda t, timeout = 300.0: {"type": "unloaded"})
    monkeypatch.setattr(o, "_drain_queue", lambda: [])

    # A generation holds _gen_lock and only releases it once cancelled -- exactly
    # what _consume_token_stream does when the worker aborts on the cancel event.
    o._gen_lock.acquire()

    def releaser():
        o._cancel_event.wait(timeout = 5)  # released only after the cancel fires
        o._gen_lock.release()

    t = threading.Thread(target = releaser)
    t.start()

    start = time.monotonic()
    ok = o.unload_model("m")
    elapsed = time.monotonic() - start
    t.join(timeout = 5)

    assert ok is True
    assert o._cancel_event.is_set(), "generation must be cancelled before the unload"
    assert {"type": "unload", "model_name": "m"} in sent
    assert o.active_model_name is None
    assert "m" not in o.models
    # It waited on the (released-after-cancel) lock, not a full generation.
    assert elapsed < 2.0


def test_unload_no_active_generation_unloads_normally(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: sent.append(cmd))
    monkeypatch.setattr(o, "_wait_response", lambda t, timeout = 300.0: {"type": "unloaded"})
    monkeypatch.setattr(o, "_drain_queue", lambda: [])

    ok = o.unload_model("m")  # _gen_lock free

    assert ok is True
    assert {"type": "unload", "model_name": "m"} in sent
    assert o.active_model_name is None
    # Lock released for the next caller.
    assert o._gen_lock.acquire(blocking = False)
    o._gen_lock.release()


def test_unload_falls_back_to_shutdown_when_generation_wont_yield(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orch_mod, "_UNLOAD_GEN_LOCK_TIMEOUT", 0.2)
    shutdown = []
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda timeout = 5: shutdown.append(timeout))
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: pytest.fail("must not send unload when wedged"))

    # A wedged worker never releases _gen_lock even after the cancel.
    o._gen_lock.acquire()

    ok = o.unload_model("m")

    assert ok is True
    assert shutdown, "should tear the subprocess down to free the GPU"
    assert o.active_model_name is None


def test_consume_token_stream_bails_when_subprocess_swapped(monkeypatch):
    # After the wedged-worker teardown a fresh load swaps _proc/_resp_queue. The
    # still-live generation thread must bail (not re-block on the new queue while
    # holding _gen_lock), so it detects the swap and returns.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_subprocess_crash_message", lambda ctx: "inference subprocess restarted")

    def read_one(timeout):
        o._proc = object()  # simulate the reload swapping the subprocess
        return None

    gen = o._consume_token_stream(read_one, lambda: None, crash_context = "generation")
    msg = next(gen)

    assert "restarted" in msg
    with pytest.raises(StopIteration):
        next(gen)
