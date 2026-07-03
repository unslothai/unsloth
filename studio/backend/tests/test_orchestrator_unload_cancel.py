# SPDX-License-Identifier: AGPL-3.0-only
"""unload_model cancels an in-flight generation instead of waiting it out.

The sequential subprocess used to queue ``unload`` behind a running ``generate``,
hanging the UI. ``unload_model`` now cancels first (the mp.Event the worker checks
each token) and takes ``_gen_lock`` before the unload round-trip.
"""
import threading
import time

import pytest

from core.inference import orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator


def _bare_orchestrator():
    """An orchestrator without the real __init__ subprocess/network."""
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._gen_lock = threading.Lock()
    o._cancel_event = threading.Event()  # stands in for the mp.Event
    o._proc = object()  # truthy so _ensure_subprocess_alive reports alive
    o._cmd_queue = object()
    o._resp_queue = object()
    o._dispatcher_thread = None
    o._unload_pending = False
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

    # A generation holds _gen_lock and releases it only once cancelled.
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
    # Waited on the released-after-cancel lock, not a full generation.
    assert elapsed < 2.0


def test_unload_no_active_generation_unloads_normally(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: sent.append(cmd))
    monkeypatch.setattr(o, "_wait_response", lambda t, timeout = 300.0: {"type": "unloaded"})
    monkeypatch.setattr(o, "_drain_queue", lambda: [])

    ok = o.unload_model("m")

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

    # A wedged worker never releases _gen_lock, even after the cancel.
    o._gen_lock.acquire()

    ok = o.unload_model("m")

    assert ok is True
    assert shutdown, "should tear the subprocess down to free the GPU"
    assert o.active_model_name is None


def test_consume_token_stream_bails_when_subprocess_swapped(monkeypatch):
    # After a wedged-worker teardown a fresh load swaps _proc/_resp_queue; the
    # still-live generation thread must detect the swap and bail, not re-block on
    # the new queue while holding _gen_lock.
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


def test_unload_pending_clears_after_unload(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: None)
    monkeypatch.setattr(o, "_wait_response", lambda t, timeout = 300.0: {"type": "unloaded"})
    monkeypatch.setattr(o, "_drain_queue", lambda: [])

    o.unload_model("m")

    # The flag must not leak past the unload, else every later generation bails.
    assert o._unload_pending is False


def test_generation_bails_when_unload_pending(monkeypatch):
    # Winning the _gen_lock handoff mid-switch must not start on the outgoing model.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    o._unload_pending = True

    out = list(o._generate_inner(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)
    # It released (or never held) the lock, so the pending unload can proceed.
    assert o._gen_lock.acquire(blocking = False)
    o._gen_lock.release()


def test_dispatched_generation_bails_when_unload_pending(monkeypatch):
    # Compare-mode bypasses _gen_lock, so it must early-out on a pending switch or
    # it enqueues a generate on the outgoing model and delays the unload.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_start_dispatcher", lambda: pytest.fail("must not start a generation mid-switch")
    )
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: pytest.fail("must not send generate mid-switch"))
    o._unload_pending = True

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)


def test_audio_input_generation_bails_when_unload_pending(monkeypatch):
    # The audio path takes _gen_lock but must also skip the outgoing model mid-switch.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: pytest.fail("must not send generate mid-switch"))
    o._unload_pending = True

    out = list(o._generate_audio_input_inner(audio_array = [0.0, 0.1]))

    assert any("unloaded" in chunk.lower() for chunk in out)
    # Lock released so the pending unload can proceed.
    assert o._gen_lock.acquire(blocking = False)
    o._gen_lock.release()


def test_audio_response_bails_when_unload_pending(monkeypatch):
    # TTS (generate_audio_response) is blocking, so it RAISES rather than starting on the
    # outgoing model mid-switch; it takes _gen_lock and must release it either way.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send audio generate mid-switch")
    )
    o._unload_pending = True

    with pytest.raises(RuntimeError, match = "unload"):
        o.generate_audio_response("hello")

    # Lock released so the pending unload can proceed.
    assert o._gen_lock.acquire(blocking = False)
    o._gen_lock.release()
