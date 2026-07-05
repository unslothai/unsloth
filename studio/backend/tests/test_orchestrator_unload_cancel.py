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
    o._drain_event = threading.Event()  # stands in for the unload-drain mp.Event
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


def test_unload_tears_down_when_compare_dispatcher_wedged(monkeypatch):
    # A wedged compare-mode generation bypasses _gen_lock, so the acquire guard
    # does not catch it and _wait_dispatcher_idle leaves the dispatcher running.
    # Proceeding to _send_cmd/_wait_response would then hang on resp_queue (the
    # dispatcher drops the request_id-less "unloaded" reply). Unload must instead
    # tear the subprocess down, like the wedged locked-generation path.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orch_mod, "_DISPATCH_IDLE_TIMEOUT", 0.2)

    # A live dispatcher whose mailbox never drains == a wedged compare-mode gen.
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {"req-1": object()}

    class _AliveThread:
        def is_alive(self):
            return True

    o._dispatcher_thread = _AliveThread()

    shutdown = []
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda timeout = 5: shutdown.append(timeout))
    monkeypatch.setattr(o, "_drain_queue", lambda: [])
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send unload with a wedged dispatcher")
    )
    monkeypatch.setattr(
        o,
        "_wait_response",
        lambda t, timeout = 300.0: pytest.fail(
            "must not wait on resp_queue with a wedged dispatcher"
        ),
    )

    # _gen_lock is free (compare mode never took it), so the acquire guard passes.
    ok = o.unload_model("m")

    assert ok is True
    assert shutdown, "should tear the subprocess down to free the GPU"
    assert o.active_model_name is None
    assert "m" not in o.models


def test_consume_token_stream_bails_when_subprocess_swapped(monkeypatch):
    # After a wedged-worker teardown a fresh load swaps _proc/_resp_queue; the
    # still-live generation thread must detect the swap and bail, not re-block on
    # the new queue while holding _gen_lock.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_subprocess_crash_message", lambda ctx: "inference subprocess restarted"
    )

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
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send generate mid-switch")
    )
    o._unload_pending = True

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)


def test_audio_input_generation_bails_when_unload_pending(monkeypatch):
    # The audio path takes _gen_lock but must also skip the outgoing model mid-switch.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send generate mid-switch")
    )
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


# ----------------------------------------------------------------------------
# Preserve unload cancels across the queue handoff (drain_event) — items #1/#4.
# ----------------------------------------------------------------------------


def test_worker_drain_skip_emits_cancelled_gen_done_when_draining():
    # The worker clears cancel_event at the start of every generate, so a cancel set
    # while a generate is still queued would be lost when it is dequeued. drain_event
    # is the durable signal: while it is set the worker skips the generate (emitting an
    # immediate gen_done so the stream/mailbox drains) instead of running it.
    import queue as _queue

    from core.inference.worker import _drain_skip_generate

    drain = threading.Event()
    rq: _queue.Queue = _queue.Queue()
    cmd = {"type": "generate", "request_id": "r1"}

    # Not draining -> run normally (do not skip, emit nothing).
    assert _drain_skip_generate(cmd, rq, drain) is False
    assert rq.empty()
    # Missing event (older worker) -> also runs normally.
    assert _drain_skip_generate(cmd, rq, None) is False
    assert rq.empty()

    # Draining -> skip and emit a cancelled gen_done for this request_id.
    drain.set()
    assert _drain_skip_generate(cmd, rq, drain) is True
    resp = rq.get_nowait()
    assert resp["type"] == "gen_done"
    assert resp["request_id"] == "r1"
    assert resp["cancelled"] is True


def test_worker_generate_branches_check_drain_before_clearing_cancel():
    # Both worker command loops (MLX fast-path + GPU) must consult the drain skip
    # before clearing cancel_event and running, so a queued generate can't clear an
    # unload-initiated cancel and run the outgoing model to completion.
    import inspect

    from core.inference import worker

    src = inspect.getsource(worker.run_inference_process)
    assert src.count("_drain_skip_generate(cmd, resp_queue, drain_event)") == 2


def test_unload_sets_drain_event_during_switch_and_clears_after(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_drain_queue", lambda: [])
    monkeypatch.setattr(o, "_wait_response", lambda t, timeout = 300.0: {"type": "unloaded"})

    seen = {}

    def record_send(cmd):
        # drain_event must be set for the whole unload round-trip so any generate the
        # worker dequeues in this window is skipped, not run.
        seen["drain_set"] = o._drain_event.is_set()

    monkeypatch.setattr(o, "_send_cmd", record_send)

    assert o.unload_model("m") is True
    assert seen.get("drain_set") is True
    # Cleared on exit so a later generation (e.g. unloading a non-active model, or a
    # reused subprocess) is not wrongly skipped.
    assert o._drain_event.is_set() is False


def test_unload_clears_drain_event_even_on_wedged_teardown(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orch_mod, "_UNLOAD_GEN_LOCK_TIMEOUT", 0.2)
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: pytest.fail("must not send when wedged"))

    # A wedged worker never releases _gen_lock; unload tears the subprocess down. The
    # real teardown nulls _drain_event, so emulate that so the finally exercises its guard.
    def fake_shutdown(timeout = 5):
        o._drain_event = None

    monkeypatch.setattr(o, "_shutdown_subprocess", fake_shutdown)
    o._gen_lock.acquire()

    assert o.unload_model("m") is True  # must not raise in the drain_event clear


# ----------------------------------------------------------------------------
# Recheck the active model after the lock wait — items #2/#3.
# ----------------------------------------------------------------------------


def test_generation_rechecks_model_after_lock_wait(monkeypatch):
    # A request passes the pre-lock active-model check, then blocks on _gen_lock while
    # an unload clears/swaps the model. Even if _unload_pending was already reset (the
    # unload's finally runs after the lock release), the under-lock active-model recheck
    # must make it bail instead of sending a generate to the wrong/unloaded backend.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not generate on a swapped/unloaded model")
    )

    reached_lock = threading.Event()
    # _wait_dispatcher_idle runs after the pre-lock check and before acquiring the lock;
    # signalling here means the generator captured the model and is about to block.
    monkeypatch.setattr(o, "_wait_dispatcher_idle", lambda: (reached_lock.set(), True)[1])

    o.active_model_name = "m"
    o._unload_pending = False
    o._gen_lock.acquire()  # stand in for an in-flight unload holding the lock

    out: list = []

    def run():
        out.extend(o._generate_inner(messages = [{"role": "user", "content": "hi"}]))

    t = threading.Thread(target = run)
    t.start()
    assert reached_lock.wait(timeout = 5)
    # Unload finished: model swapped, pending already cleared. Release the lock.
    o.active_model_name = "other"
    o._gen_lock.release()
    t.join(timeout = 5)

    assert out and any("unloaded" in chunk.lower() for chunk in out)


def test_generation_rechecks_model_when_unloaded_to_none(monkeypatch):
    # Same race, but the unload left no active model (a plain unload, not a switch).
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not generate after the model was unloaded")
    )
    reached_lock = threading.Event()
    monkeypatch.setattr(o, "_wait_dispatcher_idle", lambda: (reached_lock.set(), True)[1])

    o.active_model_name = "m"
    o._unload_pending = False
    o._gen_lock.acquire()

    out: list = []
    t = threading.Thread(
        target = lambda: out.extend(o._generate_inner(messages = [{"role": "user", "content": "hi"}]))
    )
    t.start()
    assert reached_lock.wait(timeout = 5)
    o.active_model_name = None
    o._gen_lock.release()
    t.join(timeout = 5)

    assert out and any("unloaded" in chunk.lower() for chunk in out)


# ----------------------------------------------------------------------------
# Don't unload a stale model name (worker's active-model fallback) — item #5.
# ----------------------------------------------------------------------------


def test_unload_of_stale_name_does_not_touch_active_model(monkeypatch):
    # If the named model isn't loaded (e.g. a concurrent load already swapped in a
    # different one), unload must not send a command the worker would satisfy by
    # unloading its *active* model.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send an unload for a stale model name")
    )
    o.active_model_name = "current"
    o.models = {"current": {}}

    assert o.unload_model("stale") is True
    # The active model is left intact.
    assert o.active_model_name == "current"
    assert "current" in o.models


def test_load_does_not_accumulate_stale_models_defeating_the_unload_guard(monkeypatch):
    # A load always spawns a fresh subprocess holding only the new model, so
    # self.models must mirror that instead of accumulating the previous model's name.
    # Otherwise switching A -> B (without an explicit unload of A first, e.g. the
    # concurrent-load race the stale-name guard targets) leaves 'A' in self.models,
    # so a later unload('A') passes the "model_name not in self.models" guard, reaches
    # the worker, and its absent-name fallback unloads the *active* model B.
    import types

    from utils import transformers_version as _tv

    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}

    monkeypatch.setattr(_tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([], {}))
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda *a, **k: None)
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: None)
    monkeypatch.setattr(orch_mod.time, "sleep", lambda *_a, **_k: None)

    def _load(name):
        monkeypatch.setattr(
            o,
            "_wait_response",
            lambda expected, timeout = 300.0: {
                "type": "loaded",
                "success": True,
                "model_info": {"identifier": name, "display_name": name},
            },
        )
        assert o.load_model(types.SimpleNamespace(identifier = name, gguf_variant = None)) is True

    _load("modelA")
    _load("modelB")  # switch to B without unloading A first

    # self.models mirrors the single live model; the swapped-out name is gone.
    assert o.active_model_name == "modelB"
    assert set(o.models) == {"modelB"}

    # A stale unload of the swapped-out model must not reach the worker (whose
    # absent-name fallback would unload the active model B).
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: pytest.fail("stale unload reached the worker"))
    assert o.unload_model("modelA") is True
    assert o.active_model_name == "modelB"
    assert "modelB" in o.models


def test_unload_route_serializes_with_loads_via_lifecycle_gate(monkeypatch):
    # Item #5: /unload must hold the same lifecycle gate as /load so a concurrent load
    # can't swap the backend subprocess/queues mid-unload.
    import asyncio

    import routes.inference as inference_route
    from core.inference import llama_keepwarm as kw
    from models.inference import UnloadRequest

    class _Llama:
        is_active = False
        is_loaded = False
        model_identifier = None

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference_route, "is_registered_native_path_label", lambda *a: False)

    unloaded: list = []

    class _Backend:
        active_model_name = "m"
        models = {"m": {}}

        def unload_model(self, name):
            unloaded.append(name)
            return True

    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())

    async def scenario():
        # Hold the real gate, exactly as an in-flight /load would.
        assert kw._lifecycle_lock.acquire(blocking = False)
        try:
            task = asyncio.ensure_future(
                inference_route.unload_model(UnloadRequest(model_path = "m"), "tester")
            )
            # Yield to the loop repeatedly: the route must stay blocked on the gate.
            for _ in range(10):
                await asyncio.sleep(0.01)
            assert unloaded == [], "unload ran while the lifecycle gate was held"
            assert not task.done()
        finally:
            kw._lifecycle_lock.release()
        resp = await task
        assert resp.status == "unloaded"
        assert unloaded == ["m"]

    asyncio.run(scenario())
