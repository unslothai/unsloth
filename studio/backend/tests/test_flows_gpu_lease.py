# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Player calls (/invoke) on the GPU lane: where the time goes, and how calls share the loaded chat model."""

import asyncio

import pytest

from core.flows.compiler import parse_document
from core.flows.engine import FlowEngine, GpuLease

from ._flows_helpers import doc, node, run_async


def _engine(handlers: dict) -> FlowEngine:
    eng = FlowEngine()
    eng.handlers = handlers
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    return eng


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
    gate = asyncio.Event()

    async def scene(ctx):
        await asyncio.sleep(0.05)
        ctx.text_delta("scene", "Once")
        await gate.wait()
        return {"scene": {}}

    eng = _engine({"story.next_scene": scene})
    d = _doc("story.next_scene")
    timing: dict = {}
    task = asyncio.create_task(_invoke(eng, d, "story_next_scene", timing = timing))
    await asyncio.sleep(0.2)
    gate.set()
    await task
    assert timing["gpu_wait_ms"] >= 0
    assert 40 <= timing["first_text_ms"] <= timing["total_ms"]
    assert "stop_ms" not in timing

    gate.clear()
    stopped: dict = {}
    task = asyncio.create_task(_invoke(eng, d, "story_next_scene", timing = stopped))
    await asyncio.sleep(0.2)
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

    class Engine:
        async def invoke(self, **_):
            try:
                await asyncio.sleep(60)
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
        await asyncio.sleep(0.35)
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


def _world(monkeypatch, slots: int):
    from core.flows import engine as engine_mod

    monkeypatch.setattr(engine_mod, "chat_slots", lambda: slots)
    probe = _Probe()
    eng = _engine(
        {
            "world.npc_reply": probe.handler("reply"),
            "world.npc_plan": probe.handler("plan"),
            "story.scene_art": probe.handler("image"),
        }
    )
    d = _doc("world.npc_reply", "world.npc_plan", "story.scene_art")
    return eng, d, probe


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0.01)


def test_chat_calls_on_the_loaded_model_run_side_by_side(monkeypatch):
    async def go():
        eng, d, probe = _world(monkeypatch, slots = 4)
        a = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "a"}))
        b = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "b"}))
        await probe.wait_started("a")
        await probe.wait_started("b")
        assert probe.running == {"a", "b"}
        probe.release("a")
        probe.release("b")
        await asyncio.gather(a, b)

    asyncio.run(go())


def test_one_slot_keeps_chat_calls_one_at_a_time(monkeypatch):
    async def go():
        eng, d, probe = _world(monkeypatch, slots = 1)
        a = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "a"}))
        await probe.wait_started("a")
        b = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "b"}))
        await _settle()
        assert probe.running == {"a"}
        probe.release("a")
        await probe.wait_started("b")
        probe.release("b")
        await asyncio.gather(a, b)

    asyncio.run(go())


def test_an_image_step_waits_for_chat_calls_and_holds_new_ones_back(monkeypatch):
    async def go():
        eng, d, probe = _world(monkeypatch, slots = 4)
        a = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "a"}))
        await probe.wait_started("a")
        img = asyncio.create_task(_invoke(eng, d, "story_scene_art", {"name": "img"}))
        await _settle()
        b = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "b"}))
        await _settle()
        assert probe.running == {"a"}
        probe.release("a")
        await probe.wait_started("img")
        await _settle()
        assert probe.running == {"img"}
        probe.release("img")
        await probe.wait_started("b")
        assert probe.peak["b"] == {"b"}
        probe.release("b")
        await asyncio.gather(a, img, b)

    asyncio.run(go())


def test_background_calls_leave_a_slot_for_the_player(monkeypatch):
    async def go():
        eng, d, probe = _world(monkeypatch, slots = 2)
        p1 = asyncio.create_task(_invoke(eng, d, "world_npc_plan", {"name": "p1"}))
        await probe.wait_started("p1")
        p2 = asyncio.create_task(_invoke(eng, d, "world_npc_plan", {"name": "p2"}))
        summary = asyncio.create_task(
            _invoke(eng, d, "world_npc_reply", {"name": "s", "task": "summarise"})
        )
        await _settle()
        assert probe.running == {"p1"}
        reply = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "player"}))
        await probe.wait_started("player")
        assert probe.running == {"p1", "player"}
        for name in ("p1", "player"):
            probe.release(name)
        await asyncio.gather(p1, reply)
        for _ in range(2):
            for _ in range(500):
                if probe.running:
                    break
                await asyncio.sleep(0.01)
            assert len(probe.running) == 1, probe.running
            probe.release(next(iter(probe.running)))
            await _settle()
        await asyncio.gather(p2, summary)

    asyncio.run(go())


def test_a_cancelled_chat_call_gives_its_slot_back(monkeypatch):
    async def go():
        eng, d, probe = _world(monkeypatch, slots = 4)
        a = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "a"}))
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
        eng, d, probe = _world(monkeypatch, slots = 4)
        loads: list[set[str]] = []
        tries: list[bool] = []

        async def needs_load(ctx):
            tries.append(ctx.gpu_shared)
            if ctx.gpu_shared:
                raise NeedsExclusiveGpu()
            loads.append(set(probe.running))
            return {"reply": {"text": "hi"}}

        eng.handlers["world.npc_reply"] = needs_load
        eng.handlers["world.npc_plan"] = probe.handler("plan")
        p = asyncio.create_task(_invoke(eng, d, "world_npc_plan", {"name": "p"}))
        await probe.wait_started("p")
        r = asyncio.create_task(_invoke(eng, d, "world_npc_reply", {"name": "r"}))
        await _settle()
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

    async def resident(ctx, kind):
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
        await asyncio.sleep(0.01)
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
        await asyncio.sleep(0.01)
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
        await asyncio.sleep(0.01)
        bgs = [asyncio.create_task(background(n)) for n in ("a", "b", "c")]
        await asyncio.sleep(0.01)
        switched.set()
        await asyncio.sleep(0.05)
        assert inside == ["a"]
        done.set()
        await asyncio.gather(sw, *bgs)
        assert sorted(inside) == ["a", "b", "c"]

    asyncio.run(go())
