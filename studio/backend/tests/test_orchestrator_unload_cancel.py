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
    o._dispatcher_stop = threading.Event()
    o._dispatcher_lifecycle_lock = threading.Lock()
    o._unload_pending = False
    o.active_model_name = "m"
    o.models = {"m": {}}
    o.loading_models = set()
    return o


def test_adapter_control_raises_stream_errors(monkeypatch):
    o = _bare_orchestrator()
    monkeypatch.setattr(
        o,
        "_generate_dispatched",
        lambda **_kwargs: iter([orch_mod.GenStreamError("Error: adapter failed")]),
    )

    with pytest.raises(RuntimeError, match = "adapter failed"):
        list(o.generate_with_adapter_control(use_adapter = False))

    closed = []

    def _stream(**_kwargs):
        try:
            yield "token"
            yield "late token"
        finally:
            closed.append(True)

    monkeypatch.setattr(o, "_generate_dispatched", _stream)
    generator = o.generate_with_adapter_control(use_adapter = False)
    assert next(generator) == "token"
    generator.close()
    assert closed == [True]


def test_worker_closes_cancelled_generator_before_gen_done():
    from core.inference.worker import _handle_generate

    events = []

    class _Backend:
        last_generation_stats = None

        def generate_with_adapter_control(self, **_kwargs):
            try:
                yield "token"
                yield "late token"
            finally:
                events.append("closed")

    class _Responses:
        def __init__(self):
            self.items = []

        def put(self, item):
            if item["type"] == "gen_done":
                assert events == ["closed"]
            self.items.append(item)

    responses = _Responses()
    cancel = threading.Event()
    cancel.set()
    _handle_generate(
        _Backend(),
        {"request_id": "r1", "messages": [], "use_adapter": False},
        responses,
        cancel,
    )

    assert [item["type"] for item in responses.items] == ["gen_done"]


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


def test_load_model_aborts_when_old_worker_survives_shutdown(monkeypatch):
    # A wedged worker that outlives terminate/kill makes _shutdown_subprocess return
    # False. load_model must not spawn a second worker over it (double GPU allocation +
    # the survivor's handle is lost); it aborts so the load can retry once it exits.
    import types

    from utils import transformers_version as tv

    o = _bare_orchestrator()
    o.active_model_name = "old"
    o.models = {"old": {}}
    o.loading_models = set()
    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([0], "sel"))
    monkeypatch.setattr(orch_mod.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_cancel_generation", lambda: None)
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda *a, **k: False)  # survivor
    monkeypatch.setattr(
        o, "_spawn_subprocess", lambda cfg: pytest.fail("must not spawn over a live survivor")
    )

    with pytest.raises(RuntimeError, match = "did not exit"):
        o.load_model(types.SimpleNamespace(identifier = "new", gguf_variant = None))
    # The except path cleared the loading marker and mirrors.
    assert "new" not in o.loading_models
    assert o.active_model_name is None


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


def test_load_model_aborts_when_cancelled_during_spawn(monkeypatch):
    # Stop-loading can land AFTER the pre-spawn marker recheck but while
    # _spawn_subprocess is still creating the queues/process, so cancel_load's
    # _shutdown_subprocess finds _proc not yet alive and no-ops. load_model must
    # recheck the marker once the child exists and tear the orphaned worker down,
    # instead of waiting for "loaded" and publishing a model /unload already
    # reported as unloaded (a live subprocess nothing later reaps).
    import types

    from utils import transformers_version as tv

    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}
    o.loading_models = {"m"}
    o._proc = None
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda gpu_ids, **k: ([0], "sel"))

    # The cancel lands during the spawn window: cancel_load already discarded the
    # marker, but its teardown no-oped because _proc was not alive yet.
    def spawn_then_cancel(cfg):
        o.loading_models.discard("m")

    monkeypatch.setattr(o, "_spawn_subprocess", spawn_then_cancel)

    shutdown = []
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda timeout = 5: shutdown.append(timeout))
    monkeypatch.setattr(
        o,
        "_wait_response",
        lambda t, timeout = 300.0: pytest.fail(
            "must not wait for 'loaded' after a cancel during spawn"
        ),
    )

    ok = o.load_model(types.SimpleNamespace(identifier = "m", gguf_variant = None))

    assert ok is False
    assert shutdown, "must tear the orphaned worker down"
    assert o.active_model_name is None
    assert o.models == {}
    assert "m" not in o.loading_models


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


# ----------------------------------------------------------------------------
# cancel_load clears its loading marker BEFORE tearing the subprocess down, so a
# racing off-gate load_model observes the cancel during the shutdown window.
# ----------------------------------------------------------------------------


def test_cancel_load_clears_marker_before_shutdown(monkeypatch):
    # cancel_load runs off the lifecycle gate, concurrently with a load_model that
    # rechecks the loading marker before each spawn to observe the cancel.
    # _shutdown_subprocess can block (tearing a live child down / joining the compare
    # dispatcher), so discarding the marker only AFTER it leaves a long window in which
    # that load_model reads the marker still set, passes its pre-spawn recheck, and
    # spawns + loads the model after /unload already reported it cancelled. The marker
    # (and local state) must be cleared before the teardown.
    o = _bare_orchestrator()
    o.loading_models = {"m"}
    o.active_model_name = "m"
    o.models = {"m": {}}

    at_shutdown = {}

    def record_shutdown(timeout = 5):
        at_shutdown["marker_present"] = "m" in o.loading_models
        at_shutdown["active"] = o.active_model_name
        at_shutdown["models"] = dict(o.models)

    monkeypatch.setattr(o, "_shutdown_subprocess", record_shutdown)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("cancel_load must not send a worker command")
    )

    assert o.cancel_load("m") is True
    assert at_shutdown.get("marker_present") is False, (
        "the loading marker must be cleared before _shutdown_subprocess so a concurrent "
        "load_model pre-spawn recheck observes the cancel during the shutdown window"
    )
    assert at_shutdown.get("active") is None
    assert at_shutdown.get("models") == {}
    assert "m" not in o.loading_models
    assert o.active_model_name is None
    assert o.models == {}


def test_cancel_load_reclears_state_when_racing_load_repopulates_during_teardown(monkeypatch):
    # cancel_load (off the lifecycle gate) can race a load_model whose worker already
    # queued its successful "loaded" reply. cancel_load discards the loading marker and
    # clears the local mirrors, then tears the subprocess down; but the still-running
    # load_model thread can consume that "loaded" DURING the teardown window and repopulate
    # active_model_name/models. _shutdown_subprocess nulls the queues but never touches those
    # mirrors, so without a second clear /unload reports success while the backend keeps
    # advertising a model whose worker was just killed. cancel_load must re-clear after the
    # teardown so no phantom loaded model survives.
    import types

    from utils import transformers_version as _tv

    o = _bare_orchestrator()
    o.loading_models = {"m"}
    o.active_model_name = None
    o.models = {}
    o._proc = None  # no prior subprocess -> load_model goes straight to the spawn loop

    monkeypatch.setattr(_tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([], {}))
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: None)

    parked = threading.Event()  # load_model is parked in _wait_response("loaded")
    release_loaded = threading.Event()  # cancel_load lets the load consume "loaded"
    load_done = threading.Event()

    def blocking_wait_response(expected, timeout = 300.0):
        parked.set()
        assert release_loaded.wait(timeout = 5)
        return {
            "type": "loaded",
            "success": True,
            "model_info": {"identifier": "m", "display_name": "m"},
        }

    monkeypatch.setattr(o, "_wait_response", blocking_wait_response)

    load_result: dict = {}

    def run_load():
        try:
            load_result["ok"] = o.load_model(
                types.SimpleNamespace(identifier = "m", gguf_variant = None)
            )
        except Exception as exc:  # noqa: BLE001
            load_result["exc"] = exc
        finally:
            load_done.set()

    loader = threading.Thread(target = run_load)
    loader.start()
    assert parked.wait(timeout = 5), "load_model must reach _wait_response"

    # The teardown IS the window in which the racing load repopulates the mirrors: the
    # marker is already discarded here, so release the load and wait for it to finish
    # repopulating, mirroring the 0.5s cancel-settle inside the real _shutdown_subprocess.
    def racing_shutdown(timeout = 0.5):
        release_loaded.set()
        assert load_done.wait(timeout = 5), "the racing load must repopulate during teardown"

    monkeypatch.setattr(o, "_shutdown_subprocess", racing_shutdown)

    assert o.cancel_load("m") is True
    loader.join(timeout = 5)

    # Fail-without: load_model set active_model_name/models during racing_shutdown and
    # cancel_load left them set, so the backend advertises a model whose worker was killed.
    assert o.active_model_name is None, "cancel_load must not leave a repopulated active model"
    assert o.models == {}, "cancel_load must not leave a repopulated models mirror"
    assert "m" not in o.loading_models


# ----------------------------------------------------------------------------
# A dispatched (compare-mode) request that starts the dispatcher and then bails on
# a racing unload must stop the dispatcher it started, or that orphaned dispatcher
# steals the worker's "unloaded" reply and hangs unload_model on its 300s timeout.
# ----------------------------------------------------------------------------


def test_dispatched_bail_stops_orphan_dispatcher_it_started(monkeypatch):
    # The request passes the pre-work _unload_pending check and starts the dispatcher
    # (none was running), then an unload sets _unload_pending so the under-lock recheck
    # bails. The just-started dispatcher, left running with no mailboxes, competes with
    # unload_model()'s _wait_response for the worker's "unloaded" reply off the shared
    # resp_queue and drops it as unroutable, hanging the unload until its 300s timeout.
    # The bail must stop the dispatcher it started.
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._unload_pending = False
    o._dispatcher_thread = None  # none running -> this call starts it
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)

    started = {"v": False}
    stopped = {"v": False}

    def fake_start():
        started["v"] = True
        o._dispatcher_thread = _AliveDispatcher()
        return True  # _start_dispatcher returns True for the caller that spawned it

    def fake_stop():
        stopped["v"] = True
        o._dispatcher_thread = None

    monkeypatch.setattr(o, "_start_dispatcher", fake_start)
    monkeypatch.setattr(o, "_stop_dispatcher", fake_stop)

    # An unload flips _unload_pending after the pre-work check but before registration.
    def flip(*a, **k):
        o._unload_pending = True
        return {"type": "generate", "request_id": "r1"}

    monkeypatch.setattr(o, "_build_generate_cmd", flip)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send generate after the unload flipped")
    )

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)
    assert started["v"], "this call started the dispatcher"
    assert stopped["v"], "the bail must stop the dispatcher it started (no other mailboxes)"
    assert o._mailboxes == {}


def test_dispatched_bail_keeps_dispatcher_with_other_active_mailbox(monkeypatch):
    # Guard against over-stopping: if another compare request registered a mailbox on the
    # dispatcher this call started, the bail must NOT stop it, or that request's token
    # routing dies mid-stream.
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._unload_pending = False
    o._dispatcher_thread = None
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_start_dispatcher", lambda: setattr(o, "_dispatcher_thread", _AliveDispatcher())
    )
    monkeypatch.setattr(
        o,
        "_stop_dispatcher",
        lambda: pytest.fail("must not stop a dispatcher another compare request is using"),
    )

    # A concurrent compare request registers its mailbox, then an unload flips the flag.
    def flip(*a, **k):
        o._mailboxes["other"] = object()
        o._unload_pending = True
        return {"type": "generate", "request_id": "r1"}

    monkeypatch.setattr(o, "_build_generate_cmd", flip)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send generate after the unload flipped")
    )

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)
    assert set(o._mailboxes) == {"other"}, "the other request's mailbox is untouched"


def test_dispatched_bail_keeps_preexisting_dispatcher(monkeypatch):
    # Guard: if the dispatcher was already running before this request (an earlier compare
    # request started it), a bail must not stop it even with no mailboxes now -- this
    # request did not start it and another may re-use it. Only the call that starts an
    # otherwise-idle dispatcher during the race is responsible for stopping it.
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._unload_pending = False
    o._dispatcher_thread = _AliveDispatcher()  # already running
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_start_dispatcher", lambda: None)
    monkeypatch.setattr(
        o, "_stop_dispatcher", lambda: pytest.fail("must not stop a pre-existing dispatcher")
    )

    def flip(*a, **k):
        o._unload_pending = True
        return {"type": "generate", "request_id": "r1"}

    monkeypatch.setattr(o, "_build_generate_cmd", flip)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send generate after the unload flipped")
    )

    out = list(o._generate_dispatched(messages = [{"role": "user", "content": "hi"}]))

    assert any("unloaded" in chunk.lower() for chunk in out)


# ----------------------------------------------------------------------------
# load_model rechecks the loading marker AFTER _wait_response("loaded") and
# BEFORE publishing -- item #6. cancel_load's post-teardown re-clear only wipes a
# repopulation that lands during its shutdown; a publish that lands after
# cancel_load returns survives it, so the recheck must abort the publish itself.
# ----------------------------------------------------------------------------


def test_load_model_aborts_publish_when_cancelled_after_wait_response(monkeypatch):
    # cancel_load (off the lifecycle gate) discards the loading marker BEFORE its teardown
    # and re-clears the mirrors AFTER it. A racing load_model can consume its worker's
    # already-queued "loaded" reply and reach the publish block only AFTER cancel_load has
    # fully returned -- so cancel_load's post-teardown re-clear cannot undo that publish.
    # Without a marker recheck between _wait_response("loaded") and the publish, load_model
    # advertises active_model_name/models for a model /unload already reported cancelled,
    # over a subprocess cancel_load just killed. The recheck must observe the discarded
    # marker and abort the publish.
    import types

    from utils import transformers_version as _tv

    o = _bare_orchestrator()
    o.loading_models = {"m"}
    o.active_model_name = None
    o.models = {}
    o._proc = None  # no prior subprocess -> load_model goes straight to the spawn loop

    monkeypatch.setattr(_tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([], {}))
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: None)
    # cancel_load tears the worker down; a no-op keeps the test off real subprocesses.
    monkeypatch.setattr(o, "_shutdown_subprocess", lambda timeout = 5: None)

    parked = threading.Event()  # load_model reached _wait_response("loaded")
    cancel_done = threading.Event()  # cancel_load fully returned (marker discarded + re-clear)
    load_done = threading.Event()

    def blocking_wait_response(expected, timeout = 300.0):
        parked.set()
        # Do not consume "loaded" until cancel_load has fully returned, so the publish
        # would land AFTER cancel_load's post-teardown re-clear -- the window the
        # re-clear alone cannot cover.
        assert cancel_done.wait(timeout = 5)
        return {
            "type": "loaded",
            "success": True,
            "model_info": {"identifier": "m", "display_name": "m"},
        }

    monkeypatch.setattr(o, "_wait_response", blocking_wait_response)

    load_result: dict = {}

    def run_load():
        try:
            load_result["ok"] = o.load_model(
                types.SimpleNamespace(identifier = "m", gguf_variant = None)
            )
        except Exception as exc:  # noqa: BLE001
            load_result["exc"] = exc
        finally:
            load_done.set()

    loader = threading.Thread(target = run_load)
    loader.start()
    assert parked.wait(timeout = 5), "load_model must reach _wait_response"

    # cancel_load runs to completion while the load is parked: it discards the marker and
    # re-clears the mirrors (post-teardown), then returns. Only then let the load consume
    # "loaded" and attempt to publish.
    assert o.cancel_load("m") is True
    cancel_done.set()

    loader.join(timeout = 5)
    assert load_done.is_set()

    # Fail-without: load_model published active_model_name/models for 'm' AFTER cancel_load
    # returned, advertising a cancelled model over a killed subprocess.
    assert load_result.get("ok") is False, "the cancelled load must not report success"
    assert o.active_model_name is None, "must not publish a cancelled model's active name"
    assert o.models == {}, "must not publish a cancelled model's mirror"
    assert "m" not in o.loading_models


# ----------------------------------------------------------------------------
# Concurrent compare-mode requests must not each spawn a dispatcher. Compare mode
# (_generate_dispatched) deliberately bypasses _gen_lock, so two requests can reach
# _start_dispatcher at once. Without _dispatcher_lifecycle_lock the check-then-spawn
# races: both observe no live dispatcher and each start one. The extra dispatcher is
# orphaned (self._dispatcher_thread tracks only the last) and later consumes the
# "unloaded" reply off the shared resp_queue before unload_model's _wait_response,
# hanging the unload on its 300s timeout. The lifecycle lock must serialize the
# check-then-spawn so exactly one dispatcher thread is ever created.
# ----------------------------------------------------------------------------


def test_concurrent_start_dispatcher_spawns_exactly_one():
    import queue as _queue

    o = _bare_orchestrator()
    o._resp_queue = _queue.Queue()  # real queue so the dispatcher loop blocks and stays alive
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._dispatcher_thread = None
    o._dispatcher_stop = threading.Event()
    o._dispatcher_lifecycle_lock = threading.Lock()

    n = 32
    # A barrier aligns every thread on the check-then-spawn window: without the lifecycle
    # lock several would clear the "is a dispatcher alive?" check together and each spawn one.
    barrier = threading.Barrier(n)
    results: list = []
    results_lock = threading.Lock()

    def racer():
        barrier.wait()
        started = o._start_dispatcher()
        with results_lock:
            results.append(started)

    threads = [threading.Thread(target = racer, name = f"racer-{i}") for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 5)

    try:
        # _start_dispatcher returns True only for the caller that actually spawned a thread.
        # Exactly one caller may win; every other must observe the dispatcher alive and bail.
        assert results.count(True) == 1, f"expected exactly one spawn, got {results.count(True)}"
        assert results.count(False) == n - 1
        # And exactly one live dispatcher thread exists -- no orphan racing resp_queue.
        live = [
            t for t in threading.enumerate() if t.name == "inference-dispatcher" and t.is_alive()
        ]
        assert len(live) == 1, f"expected one live dispatcher, found {len(live)}"
        assert o._dispatcher_thread is live[0]
    finally:
        o._stop_dispatcher()

    # Stop joins and clears it; no dispatcher thread must survive.
    assert o._dispatcher_thread is None
    remaining = [
        t for t in threading.enumerate() if t.name == "inference-dispatcher" and t.is_alive()
    ]
    assert remaining == [], "dispatcher must be stopped and joined"


# ----------------------------------------------------------------------------
# A compare request whose _start_dispatcher is queued behind an unload's
# _stop_dispatcher must NOT spawn a fresh dispatcher. The idle-dispatcher stop
# and the queued start both serialize on _dispatcher_lifecycle_lock; if the
# queued start spawned a new dispatcher after the stop, it would become the
# resp_queue reader and consume unload_model's "unloaded" reply (unroutable, so
# dropped) before _wait_response saw it -- hanging the unload on its 300s
# timeout. unload_model sets _unload_pending under the SAME lifecycle lock ahead
# of the stop, so _start_dispatcher observes it and refuses.
# ----------------------------------------------------------------------------


def test_start_dispatcher_refuses_while_unload_pending():
    # Direct unit guard: with an unload in progress (_unload_pending set under the
    # lifecycle lock by unload_model), _start_dispatcher must refuse and spawn nothing,
    # even though no dispatcher is currently running.
    import queue as _queue

    o = _bare_orchestrator()
    o._resp_queue = _queue.Queue()  # a spawned dispatcher would block-read here and stay alive
    o._dispatcher_thread = None
    o._dispatcher_stop = threading.Event()
    o._dispatcher_lifecycle_lock = threading.Lock()
    o._unload_pending = True

    started = o._start_dispatcher()

    assert started is False, "must not start a dispatcher while an unload is pending"
    assert o._dispatcher_thread is None, "no dispatcher thread may be created"
    live = [t for t in threading.enumerate() if t.name == "inference-dispatcher" and t.is_alive()]
    assert live == [], "no dispatcher may exist to consume the unloaded reply"


def test_start_dispatcher_resumes_after_unload_clears():
    # Guard the other direction: once the unload finishes and clears _unload_pending, a
    # later compare request must be able to start the dispatcher again (the gate must not
    # wedge). Proves the refusal above is scoped to the unload, not permanent.
    import queue as _queue

    o = _bare_orchestrator()
    o._resp_queue = _queue.Queue()
    o._dispatcher_thread = None
    o._dispatcher_stop = threading.Event()
    o._dispatcher_lifecycle_lock = threading.Lock()
    o._unload_pending = False

    try:
        assert (
            o._start_dispatcher() is True
        ), "a fresh dispatcher must start once no unload is pending"
        assert o._dispatcher_thread is not None and o._dispatcher_thread.is_alive()
    finally:
        o._stop_dispatcher()

    assert o._dispatcher_thread is None


def test_queued_start_behind_unload_stop_spawns_no_dispatcher():
    # Codex's exact ordering, forced deterministically: an unload holds
    # _dispatcher_lifecycle_lock across its _stop_dispatcher (the idle dispatcher's join
    # is gated by an event), while a compare request's _start_dispatcher is queued behind
    # it on the same lock. When the stop releases the lock the queued start must observe
    # _unload_pending (set under the lock ahead of the stop) and refuse: no fresh
    # dispatcher may be left running to steal the "unloaded" reply.
    import queue as _queue

    o = _bare_orchestrator()
    o._resp_queue = _queue.Queue()  # a spawned dispatcher would block-read here and stay alive
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._dispatcher_stop = threading.Event()
    o._dispatcher_lifecycle_lock = threading.Lock()
    o._unload_pending = False

    start_queued = threading.Event()  # release the stop's join once the start is queued behind it
    join_may_finish = threading.Event()

    class _IdleDispatcher:
        # Stand-in for the idle compare-mode dispatcher the unload stops. Its join blocks
        # until we confirm the compare _start_dispatcher is queued behind the stop, so the
        # stop provably holds _dispatcher_lifecycle_lock across that window.
        def is_alive(self):
            return True

        def join(self, timeout = None):
            assert start_queued.wait(timeout = 5), "compare start must queue behind the stop"
            assert join_may_finish.wait(timeout = 5)

    o._dispatcher_thread = _IdleDispatcher()

    def unload_side():
        # unload_model's sequence: set _unload_pending under the lifecycle lock, then stop
        # the idle dispatcher (also under the lock, via _wait_dispatcher_idle).
        with o._dispatcher_lifecycle_lock:
            o._unload_pending = True
        o._stop_dispatcher()

    started_result = {}

    def compare_side():
        started_result["v"] = o._start_dispatcher()

    u = threading.Thread(target = unload_side, name = "unload-side")
    u.start()
    # Let the unload set _unload_pending, enter _stop_dispatcher, and block in the gated join
    # while holding the lifecycle lock.
    time.sleep(0.2)

    c = threading.Thread(target = compare_side, name = "compare-side")
    c.start()
    # Let the compare _start_dispatcher block on the lifecycle lock (queued behind the stop).
    time.sleep(0.2)

    start_queued.set()  # the start is now queued behind the stop
    join_may_finish.set()  # let the stop's join complete and release the lock

    u.join(timeout = 5)
    c.join(timeout = 5)

    assert started_result.get("v") is False, "the queued start must refuse while unloading"
    assert o._dispatcher_thread is None, "the stop cleared it and the queued start spawned nothing"
    live = [t for t in threading.enumerate() if t.name == "inference-dispatcher" and t.is_alive()]
    assert live == [], "no fresh dispatcher may be left to consume the unloaded reply"
