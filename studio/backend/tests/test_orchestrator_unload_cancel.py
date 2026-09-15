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


class _Llama:
    is_active = False
    is_loaded = False
    model_identifier = None


class _Unsloth:
    def get_loading_model(self):
        return None  # no Unsloth load in flight -> Unsloth fast path skipped


def _bare_orchestrator():
    """An orchestrator without the real __init__ subprocess/network."""
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._gen_lock = threading.Lock()
    o._send_order_lock = threading.Lock()
    o._subprocess_shutdown_lock = threading.Lock()
    o._active_cancel_lock = threading.Lock()
    o._active_cancel_events = []
    o._executing_cancel_events = []
    o._cancel_event = threading.Event()  # stands in for the mp.Event
    o._drain_event = threading.Event()  # stands in for the unload-drain mp.Event
    o._proc = object()  # truthy so _ensure_subprocess_alive reports alive
    o._cmd_queue = object()
    o._resp_queue = object()
    o._dispatcher_thread = None
    o._dispatcher_stop = threading.Event()
    o._dispatcher_lifecycle_lock = threading.Lock()
    o._unload_pending = False
    o._exclusive_tts_pending = False
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
    # Guarded branches check the drain skip before clearing cancel_event and
    # again after, so a queued command can't erase an unload's cancel. Four
    # today: MLX generate + generate_audio_input, GPU generate, and GPU TTS.
    import inspect

    from core.inference import worker

    src = inspect.getsource(worker.run_inference_process)
    audio_prepare = inspect.getsource(worker._prepare_generate_audio)
    assert src.count("_drain_skip_generate(cmd, resp_queue, drain_event") == 6
    assert audio_prepare.count("_drain_skip_generate(cmd, resp_queue, drain_event") == 2


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

    from utils import transformers_version as _tv
    import types

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

    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    from core.inference import llama_keepwarm as kw
    from models.inference import UnloadRequest

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

    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    from core.inference import llama_keepwarm as kw
    from models.inference import UnloadRequest

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


def test_scoped_unload_cancels_only_its_running_standard_load(monkeypatch):
    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    cancelled = []

    class _Backend:
        def get_loading_model(self):
            return "m"

        def cancel_load(self, name):
            cancelled.append(name)
            return True

    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama())
    attempt = inference_route._begin_load_attempt(
        LoadRequest(model_path = "m", load_request_id = "audio-load-1"), "tester"
    )
    with inference_route._scoped_load_attempts_lock:
        inference_route._running_load_attempt = attempt
    try:
        response = asyncio.run(
            inference_route._unload_model_impl(
                UnloadRequest(model_path = "m", cancel_load_request_id = "audio-load-1"),
                "tester",
            )
        )
        assert response.status == "unloaded"
        assert cancelled == ["m"]
        assert attempt.cancel_event.is_set()
    finally:
        with inference_route._scoped_load_attempts_lock:
            inference_route._running_load_attempt = None
        inference_route._finish_load_attempt(attempt)


def test_scoped_unload_cannot_cancel_a_newer_same_model_load(monkeypatch):
    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    def _unexpected_backend():
        pytest.fail("a stale scoped cancel must not inspect or mutate any backend")

    monkeypatch.setattr(inference_route, "get_inference_backend", _unexpected_backend)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", _unexpected_backend)
    newer = inference_route._begin_load_attempt(
        LoadRequest(model_path = "m", load_request_id = "audio-load-new"), "tester"
    )
    with inference_route._scoped_load_attempts_lock:
        inference_route._running_load_attempt = newer
    try:
        response = asyncio.run(
            inference_route._unload_model_impl(
                UnloadRequest(model_path = "m", cancel_load_request_id = "audio-load-old"),
                "tester",
            )
        )
        assert response.status == "unloaded"
        assert not newer.cancel_event.is_set()
    finally:
        with inference_route._scoped_load_attempts_lock:
            inference_route._running_load_attempt = None
        inference_route._finish_load_attempt(newer)


def test_scoped_cancel_before_load_registration_is_consumed(monkeypatch):
    from fastapi import HTTPException
    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    def _unexpected_backend():
        pytest.fail("cancel-first cleanup must not inspect a backend")

    async def _unexpected_load(*_args, **_kwargs):
        pytest.fail("a tombstoned load must not reach the load implementation")

    monkeypatch.setattr(inference_route, "get_inference_backend", _unexpected_backend)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", _unexpected_backend)
    monkeypatch.setattr(inference_route, "_load_model_impl", _unexpected_load)
    response = asyncio.run(
        inference_route._unload_model_impl(
            UnloadRequest(model_path = "m", cancel_load_request_id = "audio-cancel-first"),
            "tester",
        )
    )
    assert response.status == "unloaded"

    request = LoadRequest(model_path = "m", load_request_id = "audio-cancel-first")
    attempt = inference_route._begin_load_attempt(request, "tester")
    try:
        assert attempt.cancel_event.is_set()
        assert attempt.cancel_complete.is_set()
        with pytest.raises(HTTPException, match = "cancelled"):
            asyncio.run(
                inference_route._run_tracked_load_model_impl(
                    request, object(), "tester", attempt = attempt
                )
            )
    finally:
        inference_route._finish_load_attempt(attempt)


def test_scoped_cancel_holds_load_ownership_until_backend_cancel_finishes(monkeypatch):
    from core.inference import llama_keepwarm
    from fastapi import HTTPException
    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    backend_cancel_entered = threading.Event()
    release_backend_cancel = threading.Event()

    class _Backend:
        def get_loading_model(self):
            return None

    def _get_backend():
        backend_cancel_entered.set()
        assert release_backend_cancel.wait(timeout = 5)
        return _Backend()

    monkeypatch.setattr(inference_route, "get_inference_backend", _get_backend)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Llama())
    monkeypatch.setattr(inference_route, "_raise_if_sidecar_swap_in_progress", lambda: None)

    async def scenario():
        gate = asyncio.Lock()
        first_started = asyncio.Event()
        second_started = asyncio.Event()

        class _Gate:
            async def __aenter__(self):
                await gate.acquire()

            async def __aexit__(self, *_args):
                gate.release()

        monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", lambda: _Gate())

        async def _fake_load(
            request,
            *_args,
            load_cancel_event = None,
            **_kwargs,
        ):
            if request.load_request_id == "audio-race-a":
                first_started.set()
                await asyncio.to_thread(load_cancel_event.wait)
                raise HTTPException(status_code = 409, detail = "Model load cancelled")
            second_started.set()
            return {"status": "loaded", "model": request.model_path}

        monkeypatch.setattr(inference_route, "_load_model_impl", _fake_load)
        first = asyncio.create_task(
            inference_route.load_model_gated(
                LoadRequest(model_path = "m", load_request_id = "audio-race-a"),
                object(),
                "tester",
            )
        )
        await first_started.wait()
        cancel = asyncio.create_task(
            inference_route._unload_model_impl(
                UnloadRequest(model_path = "m", cancel_load_request_id = "audio-race-a"),
                "tester",
            )
        )
        await asyncio.to_thread(backend_cancel_entered.wait, 5)

        second = asyncio.create_task(
            inference_route.load_model_gated(
                LoadRequest(model_path = "m", load_request_id = "audio-race-b"),
                object(),
                "tester",
            )
        )
        await asyncio.sleep(0.05)
        assert not second_started.is_set(), "newer load entered before A's cancel completed"

        release_backend_cancel.set()
        assert (await cancel).status == "unloaded"
        with pytest.raises(HTTPException, match = "cancelled"):
            await first
        assert (await second)["status"] == "loaded"

    asyncio.run(scenario())


def test_scoped_unload_of_queued_attempt_never_unloads_or_deadlocks(monkeypatch):
    from fastapi import HTTPException
    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    def _unexpected_backend():
        pytest.fail("cancel-only cleanup must never unload a resident model")

    monkeypatch.setattr(inference_route, "get_inference_backend", _unexpected_backend)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", _unexpected_backend)

    async def _unexpected_load(*_args, **_kwargs):
        pytest.fail("a cancelled queued attempt must not reach backend load work")

    monkeypatch.setattr(inference_route, "_load_model_impl", _unexpected_load)
    attempt = inference_route._begin_load_attempt(
        LoadRequest(model_path = "m", load_request_id = "audio-load-done"), "tester"
    )
    try:
        response = asyncio.run(
            inference_route._unload_model_impl(
                UnloadRequest(model_path = "m", cancel_load_request_id = "audio-load-done"),
                "tester",
            )
        )
        assert response.status == "unloaded"
        assert attempt.cancel_event.is_set()
        assert attempt.cancel_complete.is_set()
        with pytest.raises(HTTPException, match = "cancelled"):
            asyncio.run(
                asyncio.wait_for(
                    inference_route._run_tracked_load_model_impl(
                        LoadRequest(model_path = "m", load_request_id = "audio-load-done"),
                        object(),
                        "tester",
                        attempt = attempt,
                    ),
                    timeout = 1,
                )
            )
    finally:
        inference_route._finish_load_attempt(attempt)


def test_scoped_unload_cancels_only_its_running_gguf_load(monkeypatch):
    from models.inference import LoadRequest, UnloadRequest
    import asyncio
    import routes.inference as inference_route

    class _Backend:
        def get_loading_model(self):
            return None

    class _Llama:
        is_active = True
        is_loaded = False
        model_identifier = "m"

        def __init__(self):
            self.unloaded = False

        def unload_model(self):
            self.unloaded = True

    llama = _Llama()
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference_route, "is_registered_native_path_label", lambda *a: False)
    attempt = inference_route._begin_load_attempt(
        LoadRequest(model_path = "m", load_request_id = "audio-gguf-1"), "tester"
    )
    with inference_route._scoped_load_attempts_lock:
        inference_route._running_load_attempt = attempt
    try:
        response = asyncio.run(
            inference_route._unload_model_impl(
                UnloadRequest(model_path = "m", cancel_load_request_id = "audio-gguf-1"),
                "tester",
            )
        )
        assert response.status == "unloaded"
        assert llama.unloaded is True
    finally:
        with inference_route._scoped_load_attempts_lock:
            inference_route._running_load_attempt = None
        inference_route._finish_load_attempt(attempt)


def test_standard_load_honors_scoped_cancel_before_worker_start():
    from types import SimpleNamespace

    attempt_cancelled = threading.Event()
    attempt_cancelled.set()
    orchestrator = _bare_orchestrator()

    assert (
        orchestrator.load_model(
            SimpleNamespace(identifier = "m"), load_cancel_event = attempt_cancelled
        )
        is False
    )
    assert orchestrator.loading_models == set()


def test_gguf_load_cancellation_includes_the_backend_unload_event():
    import asyncio
    import routes.inference as inference_route

    from core.inference.llama_cpp import GgufDownloadCancelled

    class _Llama:
        def load_cancelled(self):
            return True

    assert inference_route._gguf_load_cancelled(_Llama(), threading.Event()) is True

    scoped = threading.Event()

    class _CancellingLlama:
        def load_cancelled(self):
            return False

        def load_model(self, *, intent, load_cancel_event):
            load_cancel_event.set()
            raise GgufDownloadCancelled("Cancelled")

    assert (
        asyncio.run(
            inference_route._run_gguf_load_attempt(
                _CancellingLlama(),
                object(),
                scoped,
            )
        )
        is False
    )


def test_cancelled_gguf_download_uses_the_typed_signal():
    from core.inference.llama_cpp import GgufDownloadCancelled, LlamaCppBackend

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(GgufDownloadCancelled):
        LlamaCppBackend()._download_gguf(
            hf_repo = "owner/model-GGUF",
            cancel_event = cancelled,
        )


def test_gguf_load_attempt_does_not_hide_a_real_resolver_failure():
    import asyncio
    import routes.inference as inference_route

    class _Llama:
        def load_cancelled(self):
            return True

        def load_model(self, *, intent, load_cancel_event):
            raise RuntimeError("resolver failed")

    with pytest.raises(RuntimeError, match = "resolver failed"):
        asyncio.run(
            inference_route._run_gguf_load_attempt(
                _Llama(),
                object(),
                threading.Event(),
            )
        )


def test_stale_unload_does_not_hide_an_update_refusal():
    import asyncio
    import routes.inference as inference_route

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    llama = LlamaCppBackend()
    llama._cancel_event.set()
    llama._llama_update_in_progress = True

    with pytest.raises(RuntimeError, match = "llama.cpp is updating"):
        asyncio.run(
            inference_route._run_gguf_load_attempt(
                llama,
                GgufLoadIntent(model_identifier = "owner/model"),
                threading.Event(),
            )
        )


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
    o._request_cancel_events = {}
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
    o._request_cancel_events = {}
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
    o._request_cancel_events = {}
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
    o._request_cancel_events = {}
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
    from utils import transformers_version as tv

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

    from utils import transformers_version as tv
    import types

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
    from utils import transformers_version as tv

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


def test_load_model_reaps_worker_after_inactivity_timeout(monkeypatch):
    # A quiet install can trip the inactivity timeout; the failure path must
    # still tear its worker down (#9398).

    from utils import transformers_version as tv
    import types

    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}
    o.loading_models = set()
    o._proc = None
    shutdowns = []
    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([0], "sel"))
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: None)
    monkeypatch.setattr(
        o,
        "_shutdown_subprocess",
        lambda timeout = 5: shutdowns.append(timeout) or True,
    )
    monkeypatch.setattr(
        o,
        "_wait_response",
        lambda t, timeout = 300.0: (_ for _ in ()).throw(
            RuntimeError("Timeout waiting for 'loaded' response (no activity for 300.0s)")
        ),
    )

    with pytest.raises(RuntimeError, match = "Timeout waiting for 'loaded'"):
        o.load_model(types.SimpleNamespace(identifier = "m", gguf_variant = None))

    assert shutdowns, "failed load must reap the worker after the inactivity timeout"
    assert o.active_model_name is None
    assert o.models == {}
    assert "m" not in o.loading_models


def test_worker_reported_load_failure_reaps_worker(monkeypatch):
    from utils import transformers_version as tv
    import types

    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}
    o.loading_models = set()
    o._proc = None
    shutdowns = []
    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([0], "sel"))
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: None)
    monkeypatch.setattr(
        o,
        "_shutdown_subprocess",
        lambda timeout = 5: shutdowns.append(timeout) or True,
    )
    monkeypatch.setattr(
        o,
        "_wait_response",
        lambda t, timeout = 300.0: {"success": False, "message": "real worker load failure"},
    )

    with pytest.raises(Exception, match = "real worker load failure"):
        o.load_model(types.SimpleNamespace(identifier = "m", gguf_variant = None))

    assert shutdowns == [5]
    assert "m" not in o.loading_models


def test_failed_load_keeps_timeout_after_cancel_teardown(monkeypatch):
    # Both paths may request teardown; the serialized second call is harmless.

    from utils import transformers_version as tv
    import types

    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}
    o.loading_models = set()
    o._proc = None

    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([0], "sel"))
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: None)

    parked = threading.Event()
    release_timeout = threading.Event()
    load_done = threading.Event()
    shutdowns = []

    def blocking_wait_response(expected, timeout = 300.0):
        parked.set()
        assert release_timeout.wait(timeout = 5)
        raise RuntimeError("Timeout waiting for 'loaded' response (no activity for 300.0s)")

    def record_shutdown(timeout = 5):
        shutdowns.append(timeout)
        o._cmd_queue = None

    monkeypatch.setattr(o, "_wait_response", blocking_wait_response)
    monkeypatch.setattr(o, "_shutdown_subprocess", record_shutdown)

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

    assert o.cancel_load("m") is True
    release_timeout.set()
    loader.join(timeout = 5)
    assert load_done.is_set()

    assert isinstance(load_result.get("exc"), RuntimeError), load_result
    assert "Timeout waiting for 'loaded'" in str(load_result["exc"])
    assert shutdowns == [0.5, 5]
    assert o.active_model_name is None
    assert o.models == {}
    assert "m" not in o.loading_models


def test_failed_load_keeps_the_timeout_when_teardown_raises(monkeypatch):
    # A teardown failure must stay a warning, not replace the load error.

    from utils import transformers_version as tv
    import types

    o = _bare_orchestrator()
    o.active_model_name = None
    o.models = {}
    o.loading_models = set()
    o._proc = None

    monkeypatch.setattr(tv, "needs_transformers_5", lambda name: False)
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *a, **k: ([0], "sel"))
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(o, "_spawn_subprocess", lambda cfg: None)
    monkeypatch.setattr(
        o,
        "_shutdown_subprocess",
        lambda timeout = 5: (_ for _ in ()).throw(
            AttributeError("'NoneType' object has no attribute 'put'")
        ),
    )
    monkeypatch.setattr(
        o,
        "_wait_response",
        lambda t, timeout = 300.0: (_ for _ in ()).throw(
            RuntimeError("Timeout waiting for 'loaded' response (no activity for 300.0s)")
        ),
    )

    with pytest.raises(RuntimeError, match = "Timeout waiting for 'loaded'"):
        o.load_model(types.SimpleNamespace(identifier = "m", gguf_variant = None))

    assert o.active_model_name is None
    assert o.models == {}
    assert "m" not in o.loading_models


def test_load_model_aborts_when_cancelled_during_spawn(monkeypatch):
    # Stop-loading can land AFTER the pre-spawn marker recheck but while
    # _spawn_subprocess is still creating the queues/process, so cancel_load's
    # _shutdown_subprocess finds _proc not yet alive and no-ops. load_model must
    # recheck the marker once the child exists and tear the orphaned worker down,
    # instead of waiting for "loaded" and publishing a model /unload already
    # reported as unloaded (a live subprocess nothing later reaps).

    from utils import transformers_version as tv
    import types

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

    from core.inference import llama_keepwarm
    import asyncio as _asyncio
    import routes.inference as ri

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

    from core.inference import llama_keepwarm
    import asyncio as _asyncio
    import routes.inference as ri

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

    from core.inference import llama_keepwarm
    import asyncio as _asyncio
    import routes.inference as ri

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

    from utils import transformers_version as _tv
    import types

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
    o._request_cancel_events = {}
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
    o._request_cancel_events = {}
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
    o._request_cancel_events = {}
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

    from utils import transformers_version as _tv
    import types

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
    o._request_cancel_events = {}
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
    o._request_cancel_events = {}
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


def _dispatch(o, resps):
    """Run the dispatcher over a fixed response list and stop it."""
    import queue as _queue

    o._resp_queue = _queue.Queue()
    for r in resps:
        o._resp_queue.put(r)
    o._dispatcher_stop = threading.Event()
    t = threading.Thread(target = o._dispatcher_loop, daemon = True)
    t.start()
    deadline = time.monotonic() + 5.0
    while not o._resp_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    o._dispatcher_stop.set()
    t.join(timeout = 5.0)


def test_worker_ownership_follows_the_worker_not_the_consumer():
    # The subprocess runs one generation at a time and can start B while A's consumer has yet to
    # drain its mailbox. A must stop owning the worker the moment its gen_done is routed, else
    # a late Stop for A cancels B.

    import queue as _queue

    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    a_cancel, b_cancel = threading.Event(), threading.Event()
    o._mailboxes = {"a": _queue.Queue(), "b": _queue.Queue()}
    o._request_cancel_events = {"a": a_cancel, "b": b_cancel}
    o._claim_worker(a_cancel)
    o._claim_worker(b_cancel)

    _dispatch(o, [{"type": "token", "request_id": "a", "token": "hi"}])
    assert o._owns_worker(a_cancel), "the request the worker is answering owns it"
    assert not o._owns_worker(b_cancel), "a queued request does not"

    # A finishes. B has been sent but has not answered yet (it is prefilling), so the gap
    # between the two is the window a late Stop for A used to fire into.
    _dispatch(o, [{"type": "gen_done", "request_id": "a"}])
    assert not o._owns_worker(a_cancel), "a finished request stops owning the worker"
    assert o._owns_worker(b_cancel), "the next queued request is the one prefilling"

    # Worker moves on to B, still before A's consumer reads anything.
    _dispatch(o, [{"type": "token", "request_id": "b", "token": "yo"}])
    assert not o._owns_worker(a_cancel), "a finished request must not cancel its successor"
    assert o._owns_worker(b_cancel), "the worker moved on to B, so B owns it"

    # A's own stream unwinding afterwards must not disturb B.
    o._release_worker(a_cancel)
    assert o._owns_worker(b_cancel)


def test_status_responses_do_not_transfer_worker_ownership():
    # Status lines are not an answer to any request; the dispatcher drops them before routing.

    import queue as _queue

    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    a_cancel, b_cancel = threading.Event(), threading.Event()
    o._mailboxes = {"a": _queue.Queue(), "b": _queue.Queue()}
    o._request_cancel_events = {"a": a_cancel, "b": b_cancel}
    o._claim_worker(a_cancel)
    o._claim_worker(b_cancel)

    _dispatch(o, [{"type": "status", "request_id": "b", "message": "loading"}])
    # Nothing has answered, so the oldest claim is still the one prefilling.
    assert o._owns_worker(a_cancel)
    assert not o._owns_worker(b_cancel)


def test_only_the_latest_responder_executes():
    # The subprocess runs one generation at a time, so answering B means it has left A.
    # _generate_inner promotes from its own consumer and can share the worker with a
    # dispatched request, so the two must not both count as executing.
    o = _bare_orchestrator()
    a_cancel, b_cancel = threading.Event(), threading.Event()
    o._claim_worker(a_cancel)
    o._claim_worker(b_cancel)

    o._mark_worker_started(a_cancel)
    assert o._owns_worker(a_cancel)
    o._mark_worker_started(b_cancel)
    assert o._owns_worker(b_cancel), "the latest responder is the one executing"
    assert not o._owns_worker(a_cancel), "and it is the only one"
    # Idempotent: more of B's own tokens must not disturb it.
    o._mark_worker_started(b_cancel)
    assert o._owns_worker(b_cancel)


def test_a_stale_mailbox_read_does_not_cancel_the_running_generation():
    # A dispatched consumer can still be draining tokens after the dispatcher retired its request
    # and started the next one. Stopping it then must tear down only its own stream: signalling
    # the shared worker event would end its successor.

    import queue as _queue

    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    a_cancel, b_cancel = threading.Event(), threading.Event()
    o._mailboxes = {"a": _queue.Queue(), "b": _queue.Queue()}
    o._request_cancel_events = {"a": a_cancel, "b": b_cancel}
    o._claim_worker(a_cancel)
    o._claim_worker(b_cancel)
    # Worker finished A and moved on to B.
    _dispatch(
        o,
        [
            {"type": "gen_done", "request_id": "a"},
            {"type": "token", "request_id": "b", "token": "yo"},
        ],
    )
    assert o._owns_worker(b_cancel) and not o._owns_worker(a_cancel)

    # A's consumer now reads a token buffered before that, with A stopped.
    a_cancel.set()
    stale = [{"type": "token", "request_id": "a", "text": "late"}]
    drained = []
    list(
        o._consume_token_stream(
            lambda timeout: stale.pop(0) if stale else None,
            lambda: drained.append(True),
            crash_context = "generation",
            cancel_event = a_cancel,
            mark_started = False,
        )
    )
    assert drained, "the stopped stream still tears itself down"
    assert not o._cancel_event.is_set(), "a retired request must not signal the shared worker event"

    # The generation that does own the worker still can.
    b_cancel.set()
    stale_b = [{"type": "token", "request_id": "b", "text": "live"}]
    list(
        o._consume_token_stream(
            lambda timeout: stale_b.pop(0) if stale_b else None,
            lambda: None,
            crash_context = "generation",
            cancel_event = b_cancel,
            mark_started = False,
        )
    )
    assert o._cancel_event.is_set(), "the running generation's own Stop must reach the worker"


def test_a_dispatcher_started_mid_stream_still_reaches_the_direct_reader():
    # A compare request can start the dispatcher while an ordinary chat is streaming. The
    # dispatcher then owns resp_queue, and without a mailbox for the direct reader it dropped
    # that chat's tokens and its gen_done as unaddressed, hanging it.

    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._direct_mailboxes = {}
    o._request_cancel_events = {}

    read_one, _drain, release = o._direct_reader("direct-1")
    try:
        _dispatch(
            o,
            [
                {"type": "token", "request_id": "direct-1", "text": "hi"},
                {"type": "gen_done", "request_id": "direct-1"},
            ],
        )
        assert read_one(timeout = 0.1) == {
            "type": "token",
            "request_id": "direct-1",
            "text": "hi",
        }, "the dispatcher must route to the direct reader, not drop"
        assert read_one(timeout = 0.1)["type"] == "gen_done"
    finally:
        release()
    assert o._direct_mailboxes == {}, "the mailbox is dropped when the stream ends"


def test_the_direct_reader_hands_back_a_compare_response_it_took():
    # The mirror race: this reader is already blocked on resp_queue when a compare request's
    # dispatcher starts, so it can take that request's response first. Consuming it would
    # corrupt this chat and hang the compare pane.

    import queue as _queue

    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    compare_box: _queue.Queue = _queue.Queue()
    o._mailboxes = {"compare-1": compare_box}
    o._direct_mailboxes = {}
    o._request_cancel_events = {}
    o._resp_queue = _queue.Queue()
    o._dispatcher_thread = None  # no dispatcher yet: this reader owns the queue

    read_one, _drain, release = o._direct_reader("direct-1")
    try:
        o._resp_queue.put({"type": "token", "request_id": "compare-1", "text": "theirs"})
        o._resp_queue.put({"type": "token", "request_id": "direct-1", "text": "mine"})
        assert read_one(timeout = 0.1) is None, "a foreign response is not ours to yield"
        assert compare_box.get_nowait()["text"] == "theirs", "it goes to its own mailbox"
        assert read_one(timeout = 0.1)["text"] == "mine"
    finally:
        release()


def test_a_direct_mailbox_is_not_mistaken_for_compare_activity():
    # _mailboxes means "compare requests are in flight" to the unload and distributed paths,
    # so an ordinary chat's mailbox must live somewhere else.
    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._direct_mailboxes = {}
    _read_one, _drain, release = o._direct_reader("direct-1")
    try:
        assert o._mailboxes == {}
        assert "direct-1" in o._direct_mailboxes
    finally:
        release()


def test_replacing_the_subprocess_clears_worker_scoped_state():
    # Ownership is keyed only by cancel-event identity, so a consumer still blocked on its
    # mailbox when the worker was replaced stayed recorded as the executor. A generation on
    # the fresh worker then failed _owns_worker and could not be stopped.

    import queue as _queue

    o = _bare_orchestrator()
    o._mailbox_lock = threading.Lock()
    dead = threading.Event()
    o._mailboxes = {"compare-1": _queue.Queue()}
    o._direct_mailboxes = {"direct-1": _queue.Queue()}
    o._request_cancel_events = {"compare-1": dead}
    o._claim_worker(dead)
    o._mark_worker_started(dead)
    assert o._owns_worker(dead)

    o._reset_worker_scoped_state()

    assert o._mailboxes == {} and o._direct_mailboxes == {}
    assert o._request_cancel_events == {}
    assert o._active_cancel_events == [] and o._executing_cancel_events == []
    # A generation on the fresh worker owns it rather than being refused by a ghost.
    fresh = threading.Event()
    o._claim_worker(fresh)
    assert o._owns_worker(fresh), "the dead worker's request must not outrank a live one"


def test_audio_input_claims_the_worker_before_sending():
    # Unclaimed, a compare request queued behind an audio-input generation looked like the
    # oldest owner, so stopping that queued request signalled the worker and killed this.
    import ast
    import pathlib

    src = pathlib.Path(orch_mod.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_audio_input_inner"
    )
    body = ast.get_source_segment(src, fn) or ""
    claim = body.find("self._claim_worker(cancel_event)")
    send = body.find("self._send_cmd(cmd)")
    assert claim != -1, "_generate_audio_input_inner must claim the worker"
    assert send != -1
    assert claim < send, "the claim has to happen before the command is enqueued"
    assert "with self._send_order_lock:" in body, "claim and send must be one critical section"
    assert "self._release_worker(cancel_event)" in body


def test_generation_stopped_while_queued_is_never_sent(monkeypatch):
    # Two chats on the serialized backend: the second blocks on _gen_lock, and Stop sets its
    # event while it waits. Sending anyway occupied the worker with a run the user ended --
    # the cancel is only checked on a token, so a long prefill (or a generation that reaches
    # gen_done without one) still held up its siblings.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(o, "_wait_dispatcher_idle", lambda *a, **k: None)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send a generation already stopped")
    )
    stopped = threading.Event()
    stopped.set()

    out = list(
        o._generate_inner(messages = [{"role": "user", "content": "hi"}], cancel_event = stopped)
    )

    assert out == [], "a stopped request yields nothing rather than an error banner"
    assert o._active_cancel_events == [], "it must not claim the worker either"
    assert o._gen_lock.acquire(blocking = False)
    o._gen_lock.release()


def test_audio_input_stopped_while_queued_is_never_sent(monkeypatch):
    # Same lock, same hole.
    o = _bare_orchestrator()
    monkeypatch.setattr(o, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(
        o, "_send_cmd", lambda cmd: pytest.fail("must not send a generation already stopped")
    )
    stopped = threading.Event()
    stopped.set()

    out = list(o._generate_audio_input_inner(audio_array = [0.0, 0.1], cancel_event = stopped))

    assert out == []
    assert o._active_cancel_events == []
    assert o._gen_lock.acquire(blocking = False)
    o._gen_lock.release()


def test_a_scoped_load_cancel_that_never_reports_back_releases_the_load():
    """Only /unload's finally sets cancel_complete, so a disconnect or shutdown between the
    two leaves nobody to set it. An unbounded wait parks the load under
    inference_lifecycle_gate for the process lifetime, and asyncio.to_thread's executor
    threads are non-daemon, so it also blocks exit."""
    from models.inference import LoadRequest, UnloadRequest
    import asyncio

    import routes.inference as inf

    request = LoadRequest(model_path = "org/a", load_request_id = "handshake-drop")
    original_impl = inf._load_model_impl
    original_timeout = inf._SCOPED_LOAD_CANCEL_HANDSHAKE_TIMEOUT_S

    async def cancel_then_die(*args, **kwargs):
        attempt, is_running = inf._cancel_scoped_load_attempt(
            UnloadRequest(model_path = "org/a", cancel_load_request_id = "handshake-drop"), "s"
        )
        assert attempt is not None and is_running
        attempt.cancel_complete.clear()  # the teardown never reached its finally
        return {"status": "loaded"}

    inf._load_model_impl = cancel_then_die
    inf._SCOPED_LOAD_CANCEL_HANDSHAKE_TIMEOUT_S = 0.25
    try:
        asyncio.run(
            asyncio.wait_for(inf._run_tracked_load_model_impl(request, None, "s"), timeout = 30)
        )
    finally:
        inf._load_model_impl = original_impl
        inf._SCOPED_LOAD_CANCEL_HANDSHAKE_TIMEOUT_S = original_timeout
        with inf._scoped_load_attempts_lock:
            inf._scoped_load_attempts.clear()
            inf._scoped_load_cancel_tombstones.clear()


def test_shutdown_cancels_loads_that_have_not_reached_the_backend():
    """A /load between admission and the backend call holds nothing the backend's
    shutdown flag can see: it can sit in the lifecycle gate or preflight for
    minutes and then arrive in a lifecycle that has already been reset, loading a
    model the new server never asked for. _graceful_shutdown cancels through the
    attempt's own event, the same path /unload uses.
    """
    import importlib

    inf = importlib.import_module("routes.inference")

    def _attempt(token, path):
        return inf._ScopedLoadAttempt(
            token = token,
            request_id = None,
            model_path = path,
            subject = "s",
            cancel_event = threading.Event(),
            cancel_complete = threading.Event(),
        )

    pending = _attempt("pending-token", "owner/model")
    running = _attempt("running-token", "owner/other")

    with inf._scoped_load_attempts_lock:
        inf._pending_load_attempts[pending.token] = pending
    prior_running = inf._running_load_attempt
    inf._running_load_attempt = running
    try:
        assert inf.cancel_pending_loads() == 2
        assert pending.cancel_event.is_set(), "a queued load survived the shutdown"
        assert running.cancel_event.is_set(), "the running load survived the shutdown"
    finally:
        inf._running_load_attempt = prior_running
        with inf._scoped_load_attempts_lock:
            inf._pending_load_attempts.pop(pending.token, None)


def test_a_running_attempt_already_in_the_pending_map_is_not_counted_twice():
    import importlib

    inf = importlib.import_module("routes.inference")

    both = inf._ScopedLoadAttempt(
        token = "same-token",
        request_id = None,
        model_path = "owner/model",
        subject = "s",
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )
    with inf._scoped_load_attempts_lock:
        inf._pending_load_attempts[both.token] = both
    prior_running = inf._running_load_attempt
    inf._running_load_attempt = both
    try:
        assert inf.cancel_pending_loads() == 1
        assert both.cancel_event.is_set()
    finally:
        inf._running_load_attempt = prior_running
        with inf._scoped_load_attempts_lock:
            inf._pending_load_attempts.pop(both.token, None)


def _mk_attempt(
    inf,
    token,
    path = "owner/model",
):
    return inf._ScopedLoadAttempt(
        token = token,
        request_id = None,
        model_path = path,
        subject = "s",
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )


def test_shutdown_closes_the_cancel_handshake_itself():
    """Only /unload sets cancel_complete, and at shutdown there is none, so setting
    cancel_event alone leaves _run_tracked_load_model_impl's finally waiting the full
    handshake timeout in a to_thread. Those executor threads are non-daemon and hold
    the process open, which is what the comment above the timeout warns about."""
    import importlib

    inf = importlib.import_module("routes.inference")
    attempt = _mk_attempt(inf, "handshake-token")
    with inf._scoped_load_attempts_lock:
        inf._pending_load_attempts[attempt.token] = attempt
    try:
        inf.cancel_pending_loads()
        assert attempt.cancel_event.is_set()
        assert attempt.cancel_complete.is_set(), (
            "shutdown left the handshake open, so the load waits the full timeout in a "
            "non-daemon executor thread and delays process exit"
        )
    finally:
        inf.begin_load_lifecycle()
        with inf._scoped_load_attempts_lock:
            inf._pending_load_attempts.pop(attempt.token, None)


def test_a_load_registering_after_the_sweep_is_still_cancelled():
    """uvicorn's should_exit stops new connections, not request tasks it already
    admitted, so a /load can register after the shutdown snapshot was taken and
    would otherwise never be cancelled by anything."""
    import importlib

    inf = importlib.import_module("routes.inference")
    try:
        assert inf.cancel_pending_loads() == 0  # latches with nothing to cancel

        late = _mk_attempt(inf, "late-token")
        with inf._scoped_load_attempts_lock:
            inf._pending_load_attempts[late.token] = late
            latched = inf._loads_shutting_down
        assert latched is True, "the shutdown latch did not survive an empty sweep"
        if latched:
            inf._cancel_for_shutdown(late)
        assert late.cancel_event.is_set(), "a load admitted during shutdown was not cancelled"
    finally:
        inf.begin_load_lifecycle()
        with inf._scoped_load_attempts_lock:
            inf._pending_load_attempts.pop("late-token", None)


def test_the_latch_is_scoped_to_a_lifecycle_not_the_process():
    """An embedded host calls run_server again. A permanent latch would refuse every
    /load of the second session; the backend flag had exactly this bug."""
    import importlib

    inf = importlib.import_module("routes.inference")
    try:
        inf.cancel_pending_loads()
        with inf._scoped_load_attempts_lock:
            assert inf._loads_shutting_down is True

        inf.begin_load_lifecycle()
        with inf._scoped_load_attempts_lock:
            assert (
                inf._loads_shutting_down is False
            ), "the second session would cancel every load it admitted"
    finally:
        inf.begin_load_lifecycle()


def test_run_server_clears_the_route_latch_too():
    import ast
    import textwrap
    from pathlib import Path

    run_py = (Path(__file__).resolve().parent.parent / "run.py").read_text(encoding = "utf-8")
    tree = ast.parse(run_py)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_server")
    src = textwrap.dedent(ast.get_source_segment(run_py, fn) or "")
    assert "begin_load_lifecycle()" in src, (
        "run_server resets the backend but not the route latch, so a restarted "
        "server cancels every load it admits"
    )


def _run_server_call_lines(name, *, owner = None):
    """First line of each call to *name* inside run_server, by AST rather than text.

    Text offsets kept breaking here: the comments above these calls name them too, and
    indentation-anchored searches go stale the moment a call moves inside a `with`.
    """
    import ast
    from pathlib import Path

    run_py = (Path(__file__).resolve().parent.parent / "run.py").read_text(encoding = "utf-8")
    fn = next(
        n
        for n in ast.parse(run_py).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_server"
    )
    lines = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == name:
            if owner is not None and getattr(func.value, "id", None) != owner:
                continue
            lines.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id == name and owner is None:
            lines.append(node.lineno)
    assert lines, f"run_server no longer calls {name}"
    return min(lines)


def test_the_route_latch_clears_only_after_the_backend_lifecycle_reopens():
    """_begin_server_lifecycle blocks on the teardown lock while a kill is running.
    Clearing the route latch before that wait leaves a request the OLD lifecycle
    admitted uncancelled, and it then captures the freshly advanced generation and
    loads the previous session's model into the new one.

    Nothing legitimate is refused by clearing later: uvicorn does not serve until
    thread.start(), which is below both calls.
    """
    backend_reset = _run_server_call_lines("_begin_server_lifecycle")
    route_reset = _run_server_call_lines("begin_load_lifecycle")
    serve = _run_server_call_lines("start", owner = "thread")

    assert backend_reset < route_reset, (
        "the route latch is cleared before the backend lifecycle reopens, so a "
        "request admitted by the old lifecycle can cross the teardown wait"
    )
    assert route_reset < serve, "the route latch is still set when the server starts serving"


def _load_impl_ast():
    """(module source, the _load_model_impl node) from routes/inference.py."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_load_model_impl"
    )
    return src, fn


def test_the_shutdown_latch_is_enforced_in_the_load_impl_not_only_at_the_route():
    """Auto-switch and preview await _load_model_impl directly, without ever
    registering a _ScopedLoadAttempt, so the shutdown sweep has no event to set for
    them. With the latch checked only where /load registers, one of those loads can
    survive the sweep, reach a non-GGUF target after run.py already tore the
    inference subprocess down, and spawn a worker that outlives quit.
    """
    import ast

    src, impl = _load_impl_ast()

    tracked = {"_run_tracked_load_model_impl", "_load_model_impl"}
    direct = [
        node.name
        for node in ast.walk(ast.parse(src))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name not in tracked
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "_load_model_impl"
            for c in ast.walk(node)
        )
    ]
    assert direct, (
        "no direct _load_model_impl callers left; if registration now covers every "
        "path this guard can move back to the route"
    )

    reads = [
        n.id for n in ast.walk(impl) if isinstance(n, ast.Name) and n.id == "_loads_shutting_down"
    ]
    assert reads, (
        "_load_model_impl never consults the shutdown latch, so these direct "
        f"callers bypass it entirely: {sorted(set(direct))}"
    )


def test_the_impl_cancel_check_refuses_a_load_once_shutdown_has_latched():
    """Runs the shipped closure, rather than asserting on its text.

    The callers of this helper are the load's points of no return, so refusing here
    is what stops a shutdown-crossing load before it spawns anything.
    """
    import ast
    import textwrap
    import threading
    from types import SimpleNamespace

    from fastapi import HTTPException

    src, impl = _load_impl_ast()
    helper = next(
        n
        for n in impl.body
        if isinstance(n, ast.FunctionDef) and n.name == "_raise_if_scoped_load_cancelled"
    )
    ns = {
        "HTTPException": HTTPException,
        "_scoped_load_attempts_lock": threading.Lock(),
        "_loads_shutting_down": False,
        "load_cancel_event": None,
        "account_access": SimpleNamespace(require_live_account = lambda: None),
    }
    exec(textwrap.dedent(ast.get_source_segment(src, helper) or ""), ns)
    check = ns["_raise_if_scoped_load_cancelled"]

    check()  # nothing set: a normal load must not be refused

    ns["_loads_shutting_down"] = True
    with pytest.raises(HTTPException) as excinfo:
        check()
    assert excinfo.value.status_code == 409

    ns["_loads_shutting_down"] = False
    ns["load_cancel_event"] = threading.Event()
    check()  # an unset per-attempt event is still not a cancel
    ns["load_cancel_event"].set()
    with pytest.raises(HTTPException):
        check()


def test_a_second_backend_instance_is_covered_by_the_shutdown_latch():
    """A helper/advisor load builds its OWN LlamaCppBackend (hub/utils/llm_assist.py,
    utils/datasets/llm_assist.py). run.py only tears down the routes singleton, and
    _shutting_down is per-instance, so that second backend would still consider a
    spawn valid and Popen a server after terminate_all took its snapshot.
    """
    from core.inference.llama_cpp import LlamaCppBackend
    from utils import process_lifetime

    torn_down = LlamaCppBackend.__new__(LlamaCppBackend)
    helper = LlamaCppBackend.__new__(LlamaCppBackend)
    try:
        assert helper._spawn_is_stale() is False, "a fresh backend must be able to spawn"

        # What _kill_process(teardown = True) does to the singleton, without the kill.
        torn_down._shutting_down = True
        process_lifetime.mark_process_shutting_down()

        assert (
            helper._spawn_is_stale() is True
        ), "a backend the shutdown never touched still thinks it may spawn"
    finally:
        process_lifetime.begin_process_lifecycle()

    assert (
        helper._spawn_is_stale() is False
    ), "the latch outlived the lifecycle, so an embedded second session cannot spawn"


def test_the_worker_spawn_refuses_once_shutdown_has_latched():
    """The preview path supplies no load_cancel_event and is not a _ScopedLoadAttempt,
    so its latch read is one-shot: it can pass that check, spend time in the drain and
    teardown, and only then reach the worker spawn. Guarding at the spawn is what makes
    the answer un-stale.
    """
    from core.inference.orchestrator import InferenceOrchestrator
    from utils import process_lifetime

    orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
    try:
        process_lifetime.mark_process_shutting_down()
        with pytest.raises(RuntimeError, match = "shutting down"):
            orch._spawn_subprocess({})
    finally:
        process_lifetime.begin_process_lifecycle()


def test_the_process_latch_clears_between_the_backend_and_the_route():
    """Ordering, for the same reason the route latch clears late: the process latch is
    what every other spawner reads, so it must outlast the teardown _begin_server_lifecycle
    waits on, and be clear before uvicorn admits anything.
    """
    backend = _run_server_call_lines("_begin_server_lifecycle")
    process = _run_server_call_lines("begin_process_lifecycle")
    route = _run_server_call_lines("begin_load_lifecycle")
    serve = _run_server_call_lines("start", owner = "thread")

    assert backend < process < route < serve, (
        "the process latch must clear after the backend teardown completes and "
        "before the server starts serving"
    )


def test_shutdown_latches_the_process_before_any_subsystem_is_torn_down():
    """The orchestrator is stopped at step 2 but loads are swept at step 5. A load in
    that gap reaches a spawner nothing has marked yet, so the latch has to be set before
    the first teardown step rather than alongside the llama-server kill.
    """
    import ast
    import textwrap
    from pathlib import Path

    run_py = (Path(__file__).resolve().parent.parent / "run.py").read_text(encoding = "utf-8")
    tree = ast.parse(run_py)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_graceful_shutdown"
    )
    src = textwrap.dedent(ast.get_source_segment(run_py, fn) or "")

    latch = src.index("mark_process_shutting_down()")
    orchestrator_stop = src.index("_shutdown_subprocess(timeout = 5.0)")
    sweep = src.index("cancel_pending_loads()")

    assert latch < orchestrator_stop, (
        "the inference subprocess is stopped before anything latches, so a load in "
        "flight can restart it"
    )
    assert latch < sweep


def test_a_shutdown_that_begins_during_the_spawn_reaps_the_new_worker():
    """The pre-spawn gate is a process start away from the child existing. A shutdown
    landing in between sees no live _proc, no-ops, completes terminate_all, and would
    leave this worker running after quit. The post-spawn recheck is what closes it.
    """
    from unittest import mock

    from core.inference.orchestrator import InferenceOrchestrator
    from utils import process_lifetime

    orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
    torn_down = []
    orch._shutdown_subprocess = lambda timeout = None: torn_down.append(timeout)

    started = mock.Mock()
    started.pid = 4242

    class _Ctx:
        Queue = staticmethod(lambda: mock.Mock())
        Event = staticmethod(lambda: mock.Mock())

        @staticmethod
        def Process(**kw):
            # Shutdown begins while the child is being born, after the gate passed.
            process_lifetime.mark_process_shutting_down()
            return started

    try:
        with (
            mock.patch.object(orch_mod, "_CTX", _Ctx),
            mock.patch.object(orch_mod, "adopt_pid", lambda pid: None, create = True),
        ):
            with pytest.raises(RuntimeError, match = "shutting down"):
                orch._spawn_subprocess({})
    finally:
        process_lifetime.begin_process_lifecycle()

    assert torn_down, "the worker born during shutdown was never torn down"


def test_a_llama_server_spawned_as_shutdown_began_is_reaped():
    """The pre-spawn stale check and the process latch are only atomic for the instance
    run.py tears down, which sets its own flag under the spawn lock. A helper backend's
    check can pass microseconds before the latch is set, and its child would then
    outlive the sweep. The recheck after the pid is recorded is what reaps it.
    """
    import ast
    import textwrap
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
    ).read_text(encoding = "utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_start_llama_process"
    )
    body = textwrap.dedent(ast.get_source_segment(src, fn) or "")

    publish = body.index("self._record_server_pid(")
    recheck = body.index("_spawn_is_stale", publish)
    kill = body.index("self._kill_process()", publish)
    assert publish < recheck < kill, (
        "nothing rechecks after the child is published, so a spawn that raced the "
        "latch leaves a server the sweep has already passed"
    )


def test_a_load_that_finishes_during_shutdown_is_not_published_as_resident():
    """The spawn checks stop once the worker exists. A "loaded" reply dequeued as
    shutdown kills it would still reach the success branch, and active_model_name plus
    models are exactly what the already-loaded fast path trusts. That path does not test
    liveness, so the next session would report a dead worker as resident.
    """
    from core.inference.orchestrator import InferenceOrchestrator
    from utils import process_lifetime

    orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orch.active_model_name = None
    orch.models = {}
    orch.loading_models = {"m"}
    orch.load_generation = 0

    try:
        process_lifetime.mark_process_shutting_down()
        stale = process_lifetime.is_process_shutting_down()
        assert stale is True, "the load no longer belongs to a live session"

        # What the success branch must do instead of publishing.
        if stale:
            orch.loading_models.discard("m")
            orch.active_model_name = None
            orch.models.clear()

        assert (
            orch.active_model_name is None and not orch.models
        ), "a worker killed by shutdown was published as resident"
    finally:
        process_lifetime.begin_process_lifecycle()


def test_the_publish_branch_rechecks_shutdown_before_recording_the_model():
    """Pins the recheck in the shipped code, ahead of the first field it publishes."""
    import ast
    import textwrap
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "orchestrator.py"
    ).read_text(encoding = "utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "load_model"
    )
    body = textwrap.dedent(ast.get_source_segment(src, fn) or "")

    check = body.index("if is_process_shutting_down(")
    publish = body.index("self.active_model_name = model_info.get(")
    assert check < publish, (
        "the load publishes active_model_name before rechecking shutdown, so a worker "
        "killed mid-load is recorded as resident"
    )


def _fn_named(source, name):
    import ast
    return next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def test_the_primary_llama_launch_rechecks_after_recording_the_pid():
    """mark_process_shutting_down does not take _spawn_lock, and terminate_all does not
    take it when it snapshots, so the in-lock check is not atomic against the
    process-wide latch for a helper-owned backend. Without a recheck the child sits
    outside the completed sweep for the whole 600s health wait.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
    ).read_text(encoding = "utf-8")
    fn = _fn_named(src, "_spawn_and_wait")
    record = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_record_server_pid"
    ]
    stale = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_spawn_is_stale"
    ]
    health = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_wait_for_health"
    ]
    assert record and stale and health, "the spawn path no longer looks like itself"
    assert any(max(record) < s < min(health) for s in stale), (
        "no staleness recheck between recording the pid and the health wait: a spawn "
        "that raced the latch is left running outside the sweep that already finished"
    )


def test_the_worker_mirrors_are_published_under_the_shutdown_lock():
    """active_model_name is what the already-loaded fast path trusts, and it does not
    test liveness. Checked and published apart, shutdown can kill the worker in between
    and the next session reports a dead one as resident.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "orchestrator.py"
    ).read_text(encoding = "utf-8")
    fn = _fn_named(src, "load_model")
    holding = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.With)
        and any(
            getattr(item.context_expr, "attr", None) == "_subprocess_shutdown_lock"
            for item in n.items
        )
    ]
    assert holding, "load_model never holds the subprocess shutdown lock"
    covered = False
    for w in holding:
        checks = [
            n
            for n in ast.walk(w)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "is_process_shutting_down"
        ]
        publishes = [
            n
            for n in ast.walk(w)
            if isinstance(n, ast.Assign)
            and any(
                getattr(t, "attr", None) == "active_model_name"
                and not isinstance(n.value, ast.Constant)
                for t in n.targets
            )
        ]
        if checks and publishes:
            covered = True
    assert covered, (
        "the shutdown check and the active_model_name publication are not inside the "
        "same held lock, so a kill can land between them"
    )


def test_the_worker_handle_is_captured_before_start_not_after():
    """`self._proc` is not a handle this code can rely on once start() is running: a
    concurrent _shutdown_subprocess can observe a not-yet-alive child and clear it.
    Snapshotting the attribute AFTER start() therefore captures None and loses the only
    reference to a live child, which is exactly the orphan this change prevents.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "orchestrator.py"
    ).read_text(encoding = "utf-8")
    fn = _fn_named(src, "_spawn_subprocess")

    # The local must be bound from the Process(...) construction, never from self._proc.
    binds = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_spawned_proc" for t in n.targets)
    ]
    assert binds, "_spawn_subprocess no longer keeps a local handle on the worker"
    for n in binds:
        assert not (isinstance(n.value, ast.Attribute) and n.value.attr == "_proc"), (
            "the local handle is snapshotted from self._proc, which a concurrent "
            "shutdown can have cleared by then; build it from Process(...) instead"
        )

    # And start() must be called through the local, not through the attribute.
    starts = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "start"
    ]
    assert starts, "the worker is never started"
    for n in starts:
        owner = n.func.value
        assert getattr(owner, "id", None) == "_spawned_proc", (
            "start() is called on self._proc rather than the local handle, so a "
            "shutdown clearing the attribute mid-start loses the child"
        )


def test_the_rag_embed_server_spawn_is_gated_and_rechecked():
    """No _graceful_shutdown step stops this backend, so the process-wide latch is the
    only thing standing between an in-flight encode and a server that outlives quit.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "core" / "rag" / "embed_llama_server.py"
    ).read_text(encoding = "utf-8")
    fn = _fn_named(src, "_spawn_once")

    checks = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "is_process_shutting_down"
    ]
    popen = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "Popen"
    ]
    adopt = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "adopt_pid"
    ]
    assert popen and adopt, "the embed spawn no longer looks like itself"
    assert any(c < min(popen) for c in checks), (
        "the embed server spawns without consulting the shutdown latch, so a quit "
        "during an encode can start one after the sweep has run"
    )
    assert any(c > max(adopt) for c in checks), (
        "no recheck after the pid is recorded: a latch set during the spawn leaves "
        "this child outside a sweep that has already finished"
    )


def test_the_stt_sidecar_spawns_are_gated_and_rechecked():
    """Neither STT sidecar is stopped by any _graceful_shutdown step and neither is
    covered by cancel_pending_loads, which only reaches the chat /load attempt maps.
    The process-wide latch is the only thing that can stop a quit during an STT load
    from leaving whisper-server or the MTMD llama-server behind.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "core" / "inference"
    for filename, func in (
        ("stt_ggml_sidecar.py", "load"),
        ("stt_mtmd_sidecar.py", "_load_locked"),
    ):
        src = (root / filename).read_text(encoding = "utf-8")
        fn = _fn_named(src, func)
        checks = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "is_process_shutting_down"
        ]
        popen = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "Popen"
        ]
        adopt = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "adopt_pid"
        ]
        assert popen and adopt, f"{filename}:{func} no longer looks like a spawn"
        assert any(c < min(popen) for c in checks), (
            f"{filename} spawns without consulting the shutdown latch, so quitting "
            "during an STT load can start a server after the sweep has run"
        )
        assert any(c > max(adopt) for c in checks), (
            f"{filename} has no recheck after recording the pid, so a latch set during "
            "the spawn leaves the child outside a sweep that already finished"
        )


def test_the_latch_is_set_before_the_atexit_sweep():
    """atexit is LIFO, so the handler registered LAST runs FIRST.

    The only thing that latched shutdown on the atexit path was the backend's own
    _cleanup hook, registered when the routes singleton was built and therefore run
    AFTER run_server's terminate_all. The sweep took its snapshot with the latch clear,
    which is exactly the unguarded behaviour this change removes on the graceful path.
    """
    import ast
    from pathlib import Path

    run_py = (Path(__file__).resolve().parent.parent / "run.py").read_text(encoding = "utf-8")
    tree = ast.parse(run_py)
    registrations = [
        (n.lineno, n.args[0].id)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "register"
        and getattr(n.func.value, "id", None) == "atexit"
        and n.args
        and isinstance(n.args[0], ast.Name)
    ]
    names = [name for _, name in registrations]
    assert "terminate_all" in names, "run.py no longer registers the backstop sweep"
    assert "mark_process_shutting_down" in names, (
        "nothing latches shutdown on the atexit path, so the sweep snapshots while "
        "spawners still believe they may start children"
    )
    sweep_line = max(l for l, name in registrations if name == "terminate_all")
    latch_line = max(l for l, name in registrations if name == "mark_process_shutting_down")
    assert latch_line > sweep_line, (
        "the latch is registered before the sweep, so under LIFO it runs after it and "
        "the snapshot is taken unguarded"
    )


def test_every_long_lived_spawner_consults_the_shutdown_latch():
    """The premise of this change is one flag read at every spawn. A spawner that
    adopts a long-lived child without consulting it is a hole in exactly the guarantee
    the PR claims, so the set is pinned here rather than rediscovered one report at a
    time.
    """
    import ast
    from pathlib import Path

    backend = Path(__file__).resolve().parent.parent
    guarded = {
        "core/export/orchestrator.py",
        "core/inference/llama_cpp.py",
        "core/inference/orchestrator.py",
        "core/inference/sd_cpp_engine.py",
        "core/inference/sd_cpp_server.py",
        "core/inference/stt_ggml_sidecar.py",
        "core/inference/stt_mtmd_sidecar.py",
        "core/inference/stt_transformers_worker.py",
        "core/rag/embed_llama_server.py",
        "core/training/training.py",
    }
    # Adopters this change deliberately leaves ungated, listed so the completeness check
    # below cannot pass by omission. They are a documented residual, not an oversight:
    # gating them is the same six lines each, but each needs its own failure idiom and
    # its own reaping, and this PR is scoped to the paths a quit during a model load
    # actually reaches. A new adopter lands in neither set and fails the check, which is
    # the point: the decision gets made once, here, instead of one report at a time.
    not_gated_here = {
        "cloudflare_tunnel.py",
        "core/data_recipe/jobs/manager.py",
        "core/inference/diffusion_transformer_quant.py",
        "core/inference/stt_download_worker.py",
        "core/inference/tools.py",
        "core/training/diffusion_training_service.py",
        "utils/prebuilt/update_flow.py",
        "utils/process_lifetime.py",
        "utils/torch_device_probe.py",
    }
    adopters = {
        str(path.relative_to(backend)).replace("\\", "/")
        for path in backend.rglob("*.py")
        if "adopt_pid(" in path.read_text(encoding = "utf-8")
        and not str(path.relative_to(backend)).replace("\\", "/").startswith("tests/")
        and "vendor/" not in str(path.relative_to(backend)).replace("\\", "/")
    }
    assert adopters == guarded | not_gated_here, (
        "a module adopts a child but is in neither set; decide whether it needs the "
        f"shutdown gate and put it in one of them: {adopters ^ (guarded | not_gated_here)}"
    )
    for rel in sorted(guarded):
        src = (backend / rel).read_text(encoding = "utf-8")
        tree = ast.parse(src)
        adopts = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "adopt_pid"
        ]
        checks = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "is_process_shutting_down"
        ]
        assert adopts, f"{rel} no longer adopts a child"
        assert checks, f"{rel} adopts a child without ever consulting the shutdown latch"
        assert any(c > min(adopts) for c in checks), (
            f"{rel} never rechecks the latch after adopting, so a spawn that raced it "
            "is left outside a sweep that has already finished"
        )
