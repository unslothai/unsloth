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
    # misses it and _send_cmd/_wait_response would hang on resp_queue. Unload must
    # instead tear the subprocess down, like the wedged locked-generation path.
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
    # unload-initiated cancel and run the outgoing model to completion. Each loop
    # checks the drain twice -- once before the clear and once after -- so a
    # drain+cancel pair that lands in the window between them is still caught.
    import inspect

    from core.inference import worker

    src = inspect.getsource(worker.run_inference_process)
    assert src.count("_drain_skip_generate(cmd, resp_queue, drain_event)") == 4


def test_worker_generate_rechecks_drain_after_clearing_cancel():
    # The exact interleaving item #3 describes: the drain check reads unset, then the
    # parent sets drain+cancel for an unload, then the worker clears cancel_event
    # (erasing that cancel). A second drain check *after* the clear catches it and
    # skips the generate instead of running the outgoing model to completion.
    import queue as _queue

    from core.inference.worker import _drain_skip_generate

    drain = threading.Event()
    cancel = threading.Event()
    rq: _queue.Queue = _queue.Queue()
    cmd = {"type": "generate", "request_id": "r1"}

    # 1. Pre-clear drain check: not draining yet -> run (no skip, no emit).
    assert _drain_skip_generate(cmd, rq, drain) is False
    assert rq.empty()

    # 2. Parent starts an unload: sets drain, then cancel (orchestrator order).
    drain.set()
    cancel.set()

    # 3. Worker clears cancel at the start of the generate -- erasing the cancel.
    cancel.clear()
    assert not cancel.is_set()

    # 4. Post-clear drain re-check catches the erased cancel and skips.
    assert _drain_skip_generate(cmd, rq, drain) is True
    resp = rq.get_nowait()
    assert resp["type"] == "gen_done" and resp["cancelled"] is True


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


def test_unload_matches_active_model_case_insensitively(monkeypatch):
    # active_model_name can differ in case from the raw model_path a client sends
    # to /unload (the load path canonicalizes casing). The stale-name guard must
    # match case-insensitively too; otherwise it no-ops the unload and leaves the
    # model resident while reporting success.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    sent = []
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: sent.append(cmd))
    monkeypatch.setattr(o, "_wait_response", lambda t, timeout = 300.0: {"type": "unloaded"})
    monkeypatch.setattr(o, "_drain_queue", lambda: [])

    o.active_model_name = "unsloth/Qwen3-4B"
    o.models = {"unsloth/Qwen3-4B": {}}

    # Client unloads with the casing it originally typed, before canonicalization.
    assert o.unload_model("unsloth/qwen3-4b") is True
    # The guard did not no-op: an unload for the canonical active model reached
    # the worker (not the raw lowercase name, so the worker matches it directly).
    assert {"type": "unload", "model_name": "unsloth/Qwen3-4B"} in sent
    # Local state is cleared for the canonical name, not left stale.
    assert o.active_model_name is None
    assert o.models == {}


def test_unload_of_stale_name_still_no_ops_after_case_insensitive_match(monkeypatch):
    # The case-insensitive match must only rescue the active model; a genuinely
    # different model name (case-insensitively too) must still no-op so the
    # worker's absent-name fallback can't tear down the active model.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send an unload for a stale model name")
    )
    o.active_model_name = "unsloth/Qwen3-4B"
    o.models = {"unsloth/Qwen3-4B": {}}

    assert o.unload_model("unsloth/Llama-3.1-8B") is True
    assert o.active_model_name == "unsloth/Qwen3-4B"
    assert "unsloth/Qwen3-4B" in o.models


def test_load_does_not_accumulate_stale_models_defeating_the_unload_guard(monkeypatch):
    # A load always spawns a fresh subprocess holding only the new model, so
    # self.models must mirror that instead of accumulating the previous model's name.
    # Otherwise switching A -> B leaves 'A' in self.models, so a later unload('A')
    # passes the "not in self.models" guard and the worker's absent-name fallback
    # unloads the *active* model B.
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


# ----------------------------------------------------------------------------
# Cancel an in-flight load OFF the lifecycle gate (Stop-loading regression).
# /load holds the gate for the whole load, so a gated /unload could never
# interrupt it; cancel_load only tears the loading subprocess down.
# ----------------------------------------------------------------------------


def test_cancel_load_terminates_loading_subprocess_and_sends_no_command(monkeypatch):
    o = _bare_orchestrator()
    o.loading_models = {"m"}
    o.active_model_name = None
    o.models = {}
    shutdown = []
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda timeout = 5: shutdown.append(timeout))
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("cancel_load must not send a worker command")
    )

    assert o.cancel_load("m") is True
    assert shutdown, "must tear the loading subprocess down"
    assert "m" not in o.loading_models
    assert o.active_model_name is None
    # A name that is not loading -> no-op, returns False so the caller takes the gate.
    assert o.cancel_load("other") is False


def test_cancel_load_matches_loading_model_case_insensitively(monkeypatch):
    o = _bare_orchestrator()
    o.loading_models = {"unsloth/Qwen3-4B"}
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda timeout = 5: None)

    assert o.cancel_load("unsloth/qwen3-4b") is True
    assert o.loading_models == set()


def test_unload_model_cancels_a_loading_model_via_cancel_load(monkeypatch):
    # unload_model still cancels an in-flight load (shared logic with cancel_load).
    o = _bare_orchestrator()
    o.loading_models = {"m"}
    o.active_model_name = None
    shutdown = []
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda timeout = 5: shutdown.append(timeout))
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send a command to cancel a load")
    )

    assert o.unload_model("m") is True
    assert shutdown
    assert "m" not in o.loading_models


def test_unload_route_cancels_in_flight_load_without_waiting_on_gate(monkeypatch):
    # The regression: /unload wrapped its whole body in the lifecycle gate, so the
    # Stop-loading button (cancelLoading -> /unload) could not interrupt a safetensors
    # load that holds the gate for its full duration. The cancel must run off-gate.
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

    cancelled: list = []

    class _Backend:
        active_model_name = None
        models: dict = {}

        def get_loading_model(self):
            return "m"

        def cancel_load(self, name):
            cancelled.append(name)
            return True

        def unload_model(self, name):
            pytest.fail("must not take the gated unload path for a still-loading model")

    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())

    async def scenario():
        # Hold the real gate, exactly as an in-flight /load would.
        assert kw._lifecycle_lock.acquire(blocking = False)
        try:
            # Even with the gate held, the loading-cancel must go through.
            resp = await inference_route.unload_model(UnloadRequest(model_path = "m"), "tester")
            assert resp.status == "unloaded"
            assert cancelled == ["m"]
        finally:
            kw._lifecycle_lock.release()

    asyncio.run(scenario())


# ----------------------------------------------------------------------------
# A dispatched (compare-mode) request that races an unload must not orphan its
# mailbox after _wait_dispatcher_idle stops the dispatcher.
# ----------------------------------------------------------------------------


def test_dispatched_bails_when_unload_flips_before_mailbox_registration(monkeypatch):
    # The request passes the pre-work _unload_pending check, then an unload sets
    # _unload_pending and _wait_dispatcher_idle stops the dispatcher (mailboxes empty)
    # before this request registers its mailbox. The recheck under _mailbox_lock must
    # make it bail, or the worker's skipped-generate reply has nothing to route it and
    # the compare stream hangs on an orphaned mailbox.
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._unload_pending = False
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_start_dispatcher", lambda: None)

    # Flip the unload flag after the pre-work check (626) but before mailbox
    # registration -- exactly the window _wait_dispatcher_idle exploits.
    def flip(*a, **k):
        o._unload_pending = True
        return {"type": "generate", "request_id": "r1"}

    monkeypatch.setattr(o, "_build_generate_cmd", flip)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send generate after the unload flipped")
    )

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)
    assert o._mailboxes == {}, "must not leave an orphaned mailbox"


# ----------------------------------------------------------------------------
# Dispatched path: bail when a cleared-pending unload swapped the model or
# tore the dispatcher down during the pre-registration window -- item #2.
# ----------------------------------------------------------------------------


class _AliveDispatcher:
    """Stand-in dispatcher thread that reports itself alive."""

    def is_alive(self):
        return True


def test_dispatched_bails_when_model_swapped_before_mailbox_registration(monkeypatch):
    # The request passes the pre-work checks, then a full unload+reload completes
    # (clearing _unload_pending) before this request registers its mailbox. The
    # under-lock recheck must notice active_model_name changed and bail, instead of
    # sending a generate that lands on the swapped-in model.
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._unload_pending = False
    o._dispatcher_thread = _AliveDispatcher()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_start_dispatcher", lambda: None)

    # Swap the active model after the pre-work check but before registration,
    # with _unload_pending already back to False (the unload finally ran).
    def swap(*a, **k):
        o.active_model_name = "other"
        return {"type": "generate", "request_id": "r1"}

    monkeypatch.setattr(o, "_build_generate_cmd", swap)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not generate on the swapped-in model")
    )

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)
    assert o._mailboxes == {}, "must not leave an orphaned mailbox"


def test_dispatched_bails_when_dispatcher_stopped_before_mailbox_registration(monkeypatch):
    # Same window, but the unload was a same-model reload so active_model_name is
    # unchanged; the give-away is that the dispatcher was stopped. Registering a
    # mailbox with no dispatcher to route the reply would hang the compare stream.
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._unload_pending = False
    o._dispatcher_thread = _AliveDispatcher()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_start_dispatcher", lambda: None)

    def stop_dispatcher(*a, **k):
        o._dispatcher_thread = None  # unload's _stop_dispatcher cleared it
        return {"type": "generate", "request_id": "r1"}

    monkeypatch.setattr(o, "_build_generate_cmd", stop_dispatcher)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not generate with the dispatcher stopped")
    )

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)
    assert o._mailboxes == {}, "must not leave an orphaned mailbox"


def test_dispatched_happy_path_registers_and_sends(monkeypatch):
    # Guard against a false bail: with the model unchanged and the dispatcher alive,
    # the recheck must let the generate through (register a mailbox and send).
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._unload_pending = False
    o._dispatcher_thread = _AliveDispatcher()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_start_dispatcher", lambda: None)
    monkeypatch.setattr(
        o, "_build_generate_cmd", lambda *a, **k: {"type": "generate", "request_id": "r1"}
    )
    sent = []
    monkeypatch.setattr(o, "_send_cmd", lambda cmd: sent.append(cmd))

    # Feed one gen_done so the consumer returns promptly.
    def fake_consume(read_mailbox, drainer, **k):
        mbox = o._mailboxes.get("r1")
        if mbox is not None:
            mbox.put({"type": "gen_done", "request_id": "r1"})
        yield ""

    monkeypatch.setattr(o, "_consume_token_stream", fake_consume)

    list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert sent, "happy path must send the generate command"
    assert o._mailboxes == {}, "mailbox popped in finally"


# ----------------------------------------------------------------------------
# load_model observes a cancel that discarded its loading marker -- item #4.
# ----------------------------------------------------------------------------


def test_load_model_aborts_when_cancelled_before_spawn(monkeypatch):
    # Stop-loading during GPU placement discards the loading marker (cancel_load) with
    # no child yet to kill. load_model must observe the removal and not spawn a worker
    # that loads the model after /unload already reported it unloaded.
    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}
    o.loading_models = set()
    o._proc = None
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda *a, **k: None)
    monkeypatch.setattr(
        o, "_spawn_subprocess", lambda cfg: pytest.fail("must not spawn a worker after a cancel")
    )

    import utils.transformers_version as tv

    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)

    # cancel_load discards the marker while we resolve GPU placement.
    def cancel_during_gpu(gpu_ids, **k):
        o.loading_models.discard("m")
        return ([0], "sel")

    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", cancel_during_gpu)

    class _Cfg:
        identifier = "m"

    ok = o.load_model(_Cfg())

    assert ok is False
    assert o.active_model_name is None
    assert o.models == {}


def test_load_model_proceeds_when_not_cancelled(monkeypatch):
    # Guard against a false abort: an uncancelled load keeps its marker and spawns.
    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}
    o.loading_models = set()
    o._proc = None
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda *a, **k: None)

    spawned = []
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: spawned.append(cfg))
    monkeypatch.setattr(
        o,
        "_wait_response",
        lambda t, timeout = 300.0: {"success": True, "model_info": {"identifier": "m"}},
    )

    import utils.transformers_version as tv

    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda gpu_ids, **k: ([0], "sel"))

    class _Cfg:
        identifier = "m"

    ok = o.load_model(_Cfg())

    assert ok is True
    assert spawned, "uncancelled load must spawn a worker"
    assert o.active_model_name == "m"


# ----------------------------------------------------------------------------
# /unload cancels a still-loading GGUF off the lifecycle gate -- item #1.
# ----------------------------------------------------------------------------


def test_unload_cancels_loading_gguf_off_gate(monkeypatch):
    # A still-loading GGUF (is_active, not is_loaded) must be cancelled off the gate:
    # /load holds the lifecycle gate for the whole load, so a gated unload would wait
    # it out. Assert the gate is never entered and unload_model() runs.
    import asyncio as _asyncio

    import routes.inference as ri
    from core.inference import llama_keepwarm

    gate_entered = {"v": False}

    class _Gate:
        async def __aenter__(self):
            gate_entered["v"] = True
            return self

        async def __aexit__(self, *a):
            return False

    class _LlamaBackend:
        is_active = True
        is_loaded = False
        model_identifier = "gguf-model"

        def __init__(self):
            self.unloaded = False

        def unload_model(self):
            self.unloaded = True

    llama = _LlamaBackend()

    class _Unsloth:
        def get_loading_model(self):
            return None  # no Unsloth load in flight -> Unsloth fast path skipped

    monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(ri, "get_inference_backend", lambda: _Unsloth())
    monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", lambda: _Gate())
    monkeypatch.setattr(llama_keepwarm, "note_model_unloaded", lambda: None)

    req = ri.UnloadRequest(model_path = "gguf-model")
    resp = _asyncio.run(ri.unload_model(req, current_subject = "s"))

    assert getattr(resp, "status", None) == "unloaded"
    assert llama.unloaded is True, "must cancel the loading GGUF via unload_model()"
    assert gate_entered["v"] is False, "must handle the loading GGUF off the lifecycle gate"


def test_unload_loaded_gguf_still_uses_gate(monkeypatch):
    # Guard: an already-loaded GGUF (is_loaded True) is NOT caught by the off-gate
    # fast path; it goes through the gate as before.
    import asyncio as _asyncio

    import routes.inference as ri
    from core.inference import llama_keepwarm

    gate_entered = {"v": False}

    class _Gate:
        async def __aenter__(self):
            gate_entered["v"] = True
            return self

        async def __aexit__(self, *a):
            return False

    class _LlamaBackend:
        is_active = True
        is_loaded = True
        model_identifier = "gguf-model"

        def __init__(self):
            self.unloaded = False

        def unload_model(self):
            self.unloaded = True

    llama = _LlamaBackend()

    class _Unsloth:
        def get_loading_model(self):
            return None

    monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(ri, "get_inference_backend", lambda: _Unsloth())
    monkeypatch.setattr(ri, "is_registered_native_path_label", lambda a, b: False)
    monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", lambda: _Gate())
    monkeypatch.setattr(llama_keepwarm, "note_model_unloaded", lambda: None)

    req = ri.UnloadRequest(model_path = "gguf-model")
    resp = _asyncio.run(ri.unload_model(req, current_subject = "s"))

    assert getattr(resp, "status", None) == "unloaded"
    assert llama.unloaded is True
    assert gate_entered["v"] is True, "loaded GGUF unload must still take the gate"


def test_unload_of_mismatched_loading_gguf_skips_off_gate_fast_path(monkeypatch):
    # A still-loading GGUF X (is_active, not is_loaded) must NOT be torn down by the
    # off-gate fast path when /unload names a DIFFERENT model Y. The single llama-server
    # can only load one GGUF at a time, so this fast path is "stop loading THIS model";
    # without a target check it fires for any in-flight GGUF and would abort an unrelated
    # load (e.g. a second tab unloading Y kills the load of X). A mismatched target must
    # fall through to the lifecycle gate (where, in production, it waits out X's /load and
    # then no-ops) instead of taking the off-gate teardown.
    import asyncio as _asyncio

    import routes.inference as ri
    from core.inference import llama_keepwarm

    gate_entered = {"v": False}

    class _Gate:
        async def __aenter__(self):
            gate_entered["v"] = True
            return self

        async def __aexit__(self, *a):
            return False

    class _LlamaBackend:
        is_active = True
        is_loaded = False
        model_identifier = "gguf-X"

        def __init__(self):
            self.unloaded = False

        def unload_model(self):
            self.unloaded = True

    llama = _LlamaBackend()

    class _Unsloth:
        def get_loading_model(self):
            return None  # no Unsloth load in flight -> Unsloth fast path skipped

    monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(ri, "get_inference_backend", lambda: _Unsloth())
    monkeypatch.setattr(ri, "is_registered_native_path_label", lambda a, b: False)
    monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", lambda: _Gate())
    monkeypatch.setattr(llama_keepwarm, "note_model_unloaded", lambda: None)

    req = ri.UnloadRequest(model_path = "gguf-Y")  # different from the loading model X
    _asyncio.run(ri.unload_model(req, current_subject = "s"))

    assert gate_entered["v"] is True, (
        "a mismatched-target unload must not use the off-gate GGUF fast path; "
        "it would cancel the wrong in-flight load"
    )
