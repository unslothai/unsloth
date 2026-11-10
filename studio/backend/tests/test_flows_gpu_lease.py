# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Player calls (/invoke) on the GPU lane: where the time goes, and how calls share the loaded chat model."""

import asyncio
import contextlib
from types import SimpleNamespace

import fastapi
import pytest

import core.inference.gpu_arbiter as arb
from core.flows import specs as specs_mod
from core.flows.compiler import parse_document
from core.flows.engine import GpuLease

from core.flows.specs import out, spec

from ._flows_helpers import chat_entry_spec, doc, make_engine, node, run_async, wait_until


def _doc(*types: str):
    parsed, errors = parse_document(doc([node(t.replace(".", "_"), t) for t in types]))
    assert not errors, errors
    return parsed


async def _invoke(
    eng,
    d,
    entry,
    state = None,
    **kw,
):
    return await eng.invoke(
        app = None,
        flow_id = "f1",
        doc = d,
        entry = entry,
        inputs = {"state": state or {}, "prompt": (state or {}).get("name")},
        **kw,
    )


@run_async
async def test_an_invoke_reports_its_gpu_wait_first_text_and_stop_time():
    gate, texted = asyncio.Event(), asyncio.Event()

    async def scene(ctx):
        await asyncio.sleep(0.05)  # the time to first text the timing must report
        ctx.text_delta("scene", "Once")
        texted.set()
        await gate.wait()
        return {"scene": {}}

    eng = make_engine({"story.next_scene": scene})
    d = _doc("story.next_scene")
    timing: dict = {}
    task = asyncio.create_task(_invoke(eng, d, "story_next_scene", timing = timing))
    await asyncio.wait_for(texted.wait(), 5)
    gate.set()
    await task
    assert timing["gpu_wait_ms"] >= 0
    assert 40 <= timing["first_text_ms"] <= timing["total_ms"]
    assert "stop_ms" not in timing

    gate.clear()
    texted.clear()
    stopped: dict = {}
    task = asyncio.create_task(_invoke(eng, d, "story_next_scene", timing = stopped))
    await asyncio.wait_for(texted.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 0 <= stopped["stop_ms"] < 1000


def test_a_dropped_background_call_stops_within_a_fraction_of_a_second(monkeypatch):
    import time

    from fastapi import HTTPException

    from routes import flows as flows_routes

    left_at: list[float] = []
    stopped_at: list[float] = []
    invoked = asyncio.Event()

    class Engine:
        async def invoke(self, **_):
            invoked.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopped_at.append(time.monotonic())
                raise

    class Request:
        app = None
        headers: dict = {}

        async def is_disconnected(self):
            return bool(left_at)

    monkeypatch.setattr(flows_routes, "get_engine", lambda: Engine())
    monkeypatch.setattr(flows_routes, "_flow_or_404", lambda fid: {"doc": {}})
    monkeypatch.setattr(flows_routes, "_hf_token", lambda request: None)
    body = flows_routes.InvokeBody(entry = "r", inputs = {})

    async def go():
        call = asyncio.create_task(flows_routes.invoke(Request(), "village", body))
        await asyncio.wait_for(invoked.wait(), 5)
        left_at.append(time.monotonic())
        with pytest.raises(HTTPException):
            await asyncio.wait_for(call, 10)

    asyncio.run(go())
    assert stopped_at[0] - left_at[0] < 0.3


class _Probe:
    def __init__(self):
        self.running: set[str] = set()
        self.peak: dict[str, set[str]] = {}
        self.started: dict[str, asyncio.Event] = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.order: list[str] = []

    def handler(self, port: str):
        async def run(ctx):
            name = (ctx.inputs.get("state") or {}).get("name") or ctx.inputs.get("prompt")
            self.order.append(name)
            self.running.add(name)
            self.peak[name] = set(self.running)
            self.started.setdefault(name, asyncio.Event()).set()
            try:
                await self.gates.setdefault(name, asyncio.Event()).wait()
            finally:
                self.running.discard(name)
            return {port: {}}

        return run

    async def wait_started(self, name: str):
        await asyncio.wait_for(self.started.setdefault(name, asyncio.Event()).wait(), 5)

    def release(self, name: str):
        self.gates.setdefault(name, asyncio.Event()).set()


def _live_chat(monkeypatch, slots: int):
    from core.flows import engine as engine_mod
    from core.flows.specs import SPECS_BY_TYPE

    monkeypatch.setattr(engine_mod, "chat_slots", lambda *_a: slots)
    monkeypatch.setitem(SPECS_BY_TYPE, "test.reply", chat_entry_spec("test.reply"))
    monkeypatch.setitem(SPECS_BY_TYPE, "test.plan", chat_entry_spec("test.plan", background = True))
    probe = _Probe()
    eng = make_engine(
        {
            "test.reply": probe.handler("reply"),
            "test.plan": probe.handler("plan"),
            "story.scene_art": probe.handler("image"),
        }
    )
    d = _doc("test.reply", "test.plan", "story.scene_art")
    return eng, d, probe


def test_the_world_live_calls_share_the_chat_model_and_planning_runs_in_the_background():
    from core.flows.specs import SPECS_BY_TYPE

    shared = {t for t, s in SPECS_BY_TYPE.items() if s.get("shares_chat")}
    background = {t for t, s in SPECS_BY_TYPE.items() if s.get("background")}
    assert shared == {
        "story.next_scene",
        "world.npc_reply",
        "world.npc_plan",
        "world.npc_chat",
        "world.describe_place",
    }
    assert background == {"world.npc_plan", "world.npc_chat"}


def _parked(lease: GpuLease) -> int:
    """Callers blocked on the lease and not yet woken: they cannot run until the lease notifies again."""
    return sum(1 for fut in lease._cond._waiters if not fut.done())


async def _until_parked(lease: GpuLease, n: int) -> None:
    await wait_until(lambda: _parked(lease) == n, 5, f"{n} callers parked on the GPU lease")


def test_chat_calls_on_the_loaded_model_run_side_by_side(monkeypatch):
    async def go():
        eng, d, probe = _live_chat(monkeypatch, slots = 4)
        a = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "a"}))
        b = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "b"}))
        await probe.wait_started("a")
        await probe.wait_started("b")
        assert probe.running == {"a", "b"}
        probe.release("a")
        probe.release("b")
        await asyncio.gather(a, b)

    asyncio.run(go())


def test_one_slot_keeps_chat_calls_one_at_a_time(monkeypatch):
    async def go():
        eng, d, probe = _live_chat(monkeypatch, slots = 1)
        a = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "a"}))
        await probe.wait_started("a")
        b = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "b"}))
        await _until_parked(eng.gpu_lock(), 1)
        assert probe.running == {"a"}
        probe.release("a")
        await probe.wait_started("b")
        probe.release("b")
        await asyncio.gather(a, b)

    asyncio.run(go())


def test_an_image_step_waits_for_chat_calls_and_holds_new_ones_back(monkeypatch):
    async def go():
        eng, d, probe = _live_chat(monkeypatch, slots = 4)
        a = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "a"}))
        await probe.wait_started("a")
        img = asyncio.create_task(_invoke(eng, d, "story_scene_art", {"name": "img"}))
        await _until_parked(eng.gpu_lock(), 1)
        b = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "b"}))
        await _until_parked(eng.gpu_lock(), 2)
        assert probe.running == {"a"}
        probe.release("a")
        await probe.wait_started("img")
        await _until_parked(eng.gpu_lock(), 1)
        assert probe.running == {"img"}
        probe.release("img")
        await probe.wait_started("b")
        assert probe.peak["b"] == {"b"}
        probe.release("b")
        await asyncio.gather(a, img, b)

    asyncio.run(go())


def test_background_calls_leave_a_slot_for_the_player(monkeypatch):
    async def go():
        eng, d, probe = _live_chat(monkeypatch, slots = 2)
        p1 = asyncio.create_task(_invoke(eng, d, "test_plan", {"name": "p1"}))
        await probe.wait_started("p1")
        p2 = asyncio.create_task(_invoke(eng, d, "test_plan", {"name": "p2"}))
        summary = asyncio.create_task(
            _invoke(eng, d, "test_reply", {"name": "s", "task": "summarise"})
        )
        await _until_parked(eng.gpu_lock(), 2)
        assert probe.running == {"p1"}
        reply = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "player"}))
        await probe.wait_started("player")
        assert probe.running == {"p1", "player"}
        for name in ("p1", "player"):
            probe.release(name)
        await asyncio.gather(p1, reply)
        for queued in (1, 0):
            # One background call at a time: the other stays parked behind it.
            await wait_until(
                lambda: probe.running and _parked(eng.gpu_lock()) == queued, 5, "the next call"
            )
            assert len(probe.running) == 1, probe.running
            probe.release(next(iter(probe.running)))
        await asyncio.gather(p2, summary)

    asyncio.run(go())


def test_a_cancelled_chat_call_gives_its_slot_back(monkeypatch):
    async def go():
        eng, d, probe = _live_chat(monkeypatch, slots = 4)
        a = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "a"}))
        await probe.wait_started("a")
        a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await a
        lease = eng.gpu_lock()
        assert not lease.locked()
        img = asyncio.create_task(_invoke(eng, d, "story_scene_art", {"name": "img"}))
        await probe.wait_started("img")
        probe.release("img")
        await img

    asyncio.run(go())


def test_a_chat_call_that_needs_a_load_waits_to_run_alone(monkeypatch):
    from core.flows.engine import NeedsExclusiveGpu
    async def go():
        eng, d, probe = _live_chat(monkeypatch, slots = 4)
        loads: list[set[str]] = []
        tries: list[bool] = []

        async def needs_load(ctx):
            tries.append(ctx.gpu_shared)
            if ctx.gpu_shared:
                raise NeedsExclusiveGpu()
            loads.append(set(probe.running))
            return {"reply": {"text": "hi"}}

        eng.handlers["test.reply"] = needs_load
        eng.handlers["test.plan"] = probe.handler("plan")
        p = asyncio.create_task(_invoke(eng, d, "test_plan", {"name": "p"}))
        await probe.wait_started("p")
        r = asyncio.create_task(_invoke(eng, d, "test_reply", {"name": "r"}))
        await _until_parked(eng.gpu_lock(), 1)
        assert loads == []
        probe.release("p")
        out = await asyncio.wait_for(r, 5)
        await p
        assert out == {"reply": {"text": "hi"}}
        assert tries == [True, False]
        assert loads == [set()]

    asyncio.run(go())


def test_ensure_model_never_loads_while_the_lane_is_shared(monkeypatch):
    from core.flows.engine import NeedsExclusiveGpu
    from core.flows.nodes import _common
    from studio_mcp import loading

    loaded: list[str] = []

    async def resident(
        ctx,
        kind,
        snap = None,
    ):
        return "other/model"

    async def load_llm(*_a, **kw):
        loaded.append(kw["model"])

    monkeypatch.setattr(_common, "resident_name", resident)
    monkeypatch.setattr(loading, "load_llm", load_llm)

    class Ctx:
        gpu_shared = True
        resident: dict = {}
        spec: dict = {}

        def set_status(self, *_a):
            pass

    with pytest.raises(NeedsExclusiveGpu):
        asyncio.run(_common._ensure_model(Ctx(), "chat", {"repo": "want/model"}, None))
    assert loaded == []


def test_a_cancelled_exclusive_waiter_lets_queued_chat_calls_in():
    async def go():
        lease = GpuLease()
        held, release = asyncio.Event(), asyncio.Event()

        async def background():
            async with lease.shared(background = True, slots = 4):
                held.set()
                await release.wait()

        async def exclusive():
            async with lease:
                pass

        bg = asyncio.create_task(background())
        await held.wait()
        ex = asyncio.create_task(exclusive())
        await _until_parked(lease, 1)
        inside = []
        both = asyncio.Event()

        async def reply(name):
            async with lease.shared(slots = 4):
                inside.append(name)
                if len(inside) == 2:
                    both.set()
                await both.wait()

        # Two calls queued: Python 3.13 wakes one waiter on a cancel by itself, 3.12 none.
        reps = [asyncio.create_task(reply(n)) for n in ("a", "b")]
        await _until_parked(lease, 3)
        assert not inside
        ex.cancel()
        await asyncio.wait_for(both.wait(), 1)
        release.set()
        await asyncio.gather(bg, *reps, return_exceptions = True)

    asyncio.run(go())


def test_chat_slots_follow_the_loaded_llama_server(monkeypatch):
    import routes.inference as inference
    from core.flows.engine import chat_slots

    class Backend:
        is_loaded = True
        effective_parallel_slots = 4

    backend = Backend()
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
    assert chat_slots() == 4
    backend.is_loaded = False
    assert chat_slots() == 1


def test_slots_are_read_again_after_a_model_switch():
    async def go():
        lease = GpuLease()
        slots = {"n": 4}
        inside = []
        switched, done = asyncio.Event(), asyncio.Event()

        async def switch():
            async with lease:
                await switched.wait()
                slots["n"] = 1

        async def background(name):
            async with lease.shared(background = True, slots = lambda: slots["n"]):
                inside.append(name)
                await done.wait()

        sw = asyncio.create_task(switch())
        await wait_until(lambda: lease._exclusive, 5, "the switch to hold the lease")
        bgs = [asyncio.create_task(background(n)) for n in ("a", "b", "c")]
        await _until_parked(lease, 3)
        switched.set()
        await wait_until(lambda: inside and _parked(lease) == 2, 5, "one call admitted")
        assert inside == ["a"]
        done.set()
        await asyncio.gather(sw, *bgs)
        assert sorted(inside) == ["a", "b", "c"]

    asyncio.run(go())


# The operation lease (gpu_arbiter): benchmarks and evals hold the GPU against other residency changes.


_HELD: list = []


def _lease_elsewhere(label: str = "Benchmark Model"):
    """Hold the operation lease for another holder until the test ends (see _release_held_leases)."""
    lease = arb.operation_lease("bench", "run-1", 60, label = label)
    lease.__enter__()
    arb._lease_holder.set(None)
    _HELD.append(lease)
    return lease


@pytest.fixture(autouse = True)
def _release_held_leases():
    """Close leases taken by _lease_elsewhere here, not when the garbage collector finds them in another context."""
    yield
    while _HELD:
        with contextlib.suppress(Exception):
            _HELD.pop().__exit__(None, None, None)


def _add_spec(monkeypatch, found: dict) -> None:
    monkeypatch.setitem(specs_mod.SPECS_BY_TYPE, found["type"], found)


def test_an_exclusive_node_always_runs_on_the_gpu_lane():
    from core.flows.specs import spec
    assert spec("t.x", "X", "c", "d", exclusive_gpu = True)["lane"] == "gpu"
    assert spec("t.y", "Y", "c", "d")["lane"] == "cpu"


def test_a_lease_refuses_other_loads_and_training_but_not_its_holder(no_gpu_lease, monkeypatch):
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: None)
    assert arb.lease_refusal() is None
    arb.acquire_for(arb.CHAT, lambda: None)

    async def holder():
        with arb.operation_lease("bench", "run-1", 60, label = "Benchmark Model"):
            arb.acquire_for(arb.CHAT, lambda: None)
            await asyncio.to_thread(arb.acquire_for, arb.CHAT, lambda: None)
            other = asyncio.get_running_loop()
            refusal = await other.run_in_executor(None, arb.lease_refusal)
            return refusal

    refusal = asyncio.run(holder())
    assert isinstance(refusal, arb.GpuLeaseHeldError)
    assert "Benchmark Model is using the GPU" in str(refusal)
    assert arb.lease_refusal() is None


def test_a_refused_load_keeps_the_load_generation(no_gpu_lease, monkeypatch):
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    arb.acquire_for(arb.CHAT, lambda: None)
    before = arb.load_count()

    def refuse():
        raise RuntimeError("a load is already in progress")

    with pytest.raises(RuntimeError):
        arb.acquire_for(arb.CHAT, refuse)
    assert arb.load_count() == before


def test_while_leased_other_callers_get_409_and_plain_reasserts_pass(no_gpu_lease, monkeypatch):
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: None)
    arb.acquire_for(arb.CHAT, lambda: None)
    _lease_elsewhere()
    try:
        with pytest.raises(arb.GpuLeaseHeldError):
            arb.acquire_for(arb.DIFFUSION, lambda: None)
        with pytest.raises(arb.GpuLeaseHeldError):
            arb.acquire_for(arb.CHAT, lambda: None)
        arb.acquire_for(arb.CHAT)
        http = arb.lease_refusal().as_http_exception()
        assert http.status_code == 409 and http.detail["error"] == "gpu_leased"
        with pytest.raises(arb.GpuLeaseHeldError):
            with arb.operation_lease("eval", "run-2", 60):
                pass
    finally:
        arb._lease = None


def test_a_reassert_under_a_lease_never_evicts_the_leased_model(no_gpu_lease, monkeypatch):
    from core.inference import gpu_residency

    made_room, released = [], []
    monkeypatch.setattr(gpu_residency, "coreside_locked", lambda *a, **k: made_room.append(a))
    monkeypatch.setattr(arb, "_release_mask_segmenter", lambda *a, **k: released.append("mask"))
    monkeypatch.setattr(
        arb, "_release_idle_video_servers", lambda *a, **k: released.append("video")
    )
    monkeypatch.setattr(arb, "_owner", arb.CHAT)
    monkeypatch.setattr(arb, "_residents", {arb.DIFFUSION: None, arb.CHAT: None})
    arb.acquire_for(arb.CHAT, policy = "fit")
    assert len(made_room) == 1
    released.clear()
    _lease_elsewhere("Benchmark Image")
    try:
        arb.acquire_for(arb.CHAT, policy = "fit")
        arb.acquire_for(arb.CHAT)
    finally:
        arb._lease = None
    assert len(made_room) == 1 and released == []
    assert arb.DIFFUSION in arb._residents


def test_an_abandoned_lease_expires(no_gpu_lease, monkeypatch):
    import time

    lease = arb.operation_lease("bench", "run-1", 1)
    lease.__enter__()
    arb._lease_holder.set(None)
    assert arb.lease_refusal() is not None
    arb._lease["expires"] = time.monotonic() - 1
    assert arb.lease_refusal() is None and not arb.lease_held()


def test_a_lease_waits_for_a_residency_change_already_under_way(no_gpu_lease):
    with arb._lock:
        with pytest.raises(arb.GpuLeaseHeldError):
            with arb.operation_lease("bench", "run-1", 60):
                pass
    assert arb._lease is None
    with arb.operation_lease("bench", "run-1", 60):
        assert arb._lease is not None


def _measure_spec():
    return spec(
        "test.measure",
        "Measure",
        "logic",
        "Holds the GPU.",
        outputs = [out("value", "text")],
        lane = "gpu",
        cacheable = False,
        exclusive_gpu = True,
    )


@run_async
async def test_an_exclusive_node_holds_the_lease_and_its_cancel_hook_still_acts_as_holder(
    no_gpu_lease, monkeypatch
):
    _add_spec(monkeypatch, _measure_spec())
    seen = {}
    started = asyncio.Event()

    async def measure(ctx):
        seen["inside"] = arb.lease_refusal() is None and arb.lease_held()

        async def restore():
            seen["hook"] = arb.lease_refusal() is None

        ctx.on_cancel(restore)
        started.set()
        await asyncio.Event().wait()  # held until Stop cancels it
        return {"value": "never"}

    eng = make_engine({"test.measure": measure})
    parsed, _ = parse_document(doc([node("m", "test.measure")]))
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed)
    await asyncio.wait_for(started.wait(), 5)
    assert isinstance(arb.lease_refusal(), arb.GpuLeaseHeldError)
    await eng.cancel(rs.id)
    await asyncio.wait_for(rs.task, 20)
    assert seen == {"inside": True, "hook": True}
    assert rs.status == "cancelled"
    assert not arb.lease_held()


@run_async
async def test_an_exclusive_node_waits_for_another_lease(no_gpu_lease, monkeypatch):
    _add_spec(monkeypatch, _measure_spec())
    monkeypatch.setattr("core.flows.engine.LEASE_RETRY_S", 0.05)

    async def measure(ctx):
        return {"value": "ran"}

    other = arb.operation_lease("eval", "someone", 60, label = "Evaluate Model")
    other.__enter__()
    arb._lease_holder.set(None)
    eng = make_engine({"test.measure": measure})
    parsed, _ = parse_document(doc([node("m", "test.measure")]))
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed)
    await wait_until(
        lambda: rs.nodes["m"].status == "waiting", 10, "the step to wait for the lease"
    )
    assert "Evaluate Model is using the GPU" in (rs.nodes["m"].message or "")
    arb._lease = None
    await asyncio.wait_for(rs.task, 20)
    assert rs.nodes["m"].outputs == {"value": "ran"}


def test_unloads_and_training_teardown_respect_a_lease_taken_later(no_gpu_lease):
    import fastapi
    from routes import inference, training

    _lease_elsewhere()
    try:
        with pytest.raises(fastapi.HTTPException) as err:
            inference._refuse_while_leased()
        assert err.value.status_code == 409
        with pytest.raises(arb.GpuLeaseHeldError):
            training._free_gpu_for_diffusion_training()
        assert arb.GpuLeaseHeldError.blocks_training is True
    finally:
        arb._lease = None
    with arb.operation_lease("bench", "run-1", 60):
        inference._refuse_while_leased()


def test_every_unload_path_shares_the_lease_guard(no_gpu_lease):
    import fastapi
    from models.inference import UnloadRequest
    from routes import inference

    _lease_elsewhere()
    try:
        with pytest.raises(fastapi.HTTPException) as err:
            asyncio.run(inference._unload_model_impl(UnloadRequest(model_path = "x"), "owner"))
        assert err.value.status_code == 409
    finally:
        arb._lease = None


@pytest.fixture
def training_start(monkeypatch):
    """start_training with the admission and the trainer faked: records what got past the start checks."""
    from routes import training

    seen = {"admitted": [], "spawned": [], "lease_before_spawn": False, "leases": []}

    class Backend:
        current_job_id = ""

        def is_training_active(self):
            return False

        def start_training(self, before_spawn, **kw):
            if seen["lease_before_spawn"]:
                seen["leases"].append(_lease_elsewhere())
            before_spawn()
            seen["spawned"].append(kw)
            return True

    async def admit(request, subject, via_api_key):
        seen["admitted"].append(request)
        return SimpleNamespace(training_kwargs = {}, hf_token = None, resume_run = None)

    monkeypatch.setattr(training, "get_training_backend", lambda: Backend())
    monkeypatch.setattr(training, "_admit_training_request", admit)
    monkeypatch.setattr(training, "_diffusion_training_active", lambda: False)
    monkeypatch.setattr(training, "_batch_members_for_start", lambda kwargs: None)
    return seen


def _training_request():
    from models.training import TrainingStartRequest
    return TrainingStartRequest(
        model_name = "unsloth/Qwen3-0.6B", training_type = "LoRA/QLoRA", format_type = "auto"
    )


def _start_training(kind: str):
    from models.training import DiffusionTrainingStartRequest
    from routes import training

    if kind == "start_training":
        return training.start_training(_training_request(), "owner", False)
    if kind == "start_training_batch":
        return training.start_training_batch(
            [_training_request(), _training_request()], "owner", False
        )
    body = DiffusionTrainingStartRequest(
        base_model = "x", data_dir = "/nonexistent", output_dir = "/nonexistent"
    )
    return training.start_diffusion_training(body, "owner", False)


@pytest.mark.parametrize(
    "kind", ["start_training", "start_training_batch", "start_diffusion_training"]
)
def test_training_start_is_refused_while_another_holder_has_the_lease(
    no_gpu_lease, training_start, kind
):
    _lease_elsewhere()
    with pytest.raises(fastapi.HTTPException) as err:
        asyncio.run(_start_training(kind))
    assert err.value.status_code == 409
    assert training_start["admitted"] == [] and training_start["spawned"] == []


def test_training_teardown_refuses_a_lease_taken_after_the_start_checks(
    no_gpu_lease, training_start
):
    training_start["lease_before_spawn"] = True
    with pytest.raises(fastapi.HTTPException) as err:
        asyncio.run(_start_training("start_training"))
    assert err.value.status_code == 409
    assert len(training_start["admitted"]) == 1 and training_start["spawned"] == []


@pytest.mark.parametrize("route", ["video", "diffusion"])
def test_media_unloads_are_refused_under_another_holders_lease(no_gpu_lease, route):
    from routes import inference, video

    unload = video.unload_video_model if route == "video" else inference.unload_diffusion_model
    _lease_elsewhere()
    with pytest.raises(fastapi.HTTPException) as err:
        asyncio.run(unload(None, "owner", False))
    assert err.value.status_code == 409
