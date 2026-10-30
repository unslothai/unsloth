# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model residency across a run: reuse, load-ahead during CPU steps, cache-hit skips, unload at run end."""

import asyncio
import time

import pytest

from core.flows import engine as engine_mod
from core.flows.compiler import parse_document
from core.flows.engine import FlowEngine
from core.flows.nodes import _common
from studio_mcp import loading

from ._flows_helpers import doc, edge, node, run_async


@pytest.fixture
def studio(monkeypatch):
    state = {"slots": {}, "loads": [], "unloads": []}

    async def resident_slots(caller):
        return {k: list(v) for k, v in state["slots"].items()}

    async def ensure_model(
        ctx,
        kind,
        choice,
        options = None,
    ):
        repo = choice.get("repo") if isinstance(choice, dict) else None
        if not repo:
            return None
        held = state["slots"].get("llm") or []
        if repo not in held if state.get("keep") else held != [repo]:
            await asyncio.sleep(0.2)  # the load itself
            state["loads"].append((repo, time.monotonic()))
            state["slots"]["llm"] = [*held, repo] if state.get("keep") else [repo]
            ctx.note_loaded("llm", repo)
        ctx.resident["llm"] = repo
        return {"repo": repo}

    async def unload_llm(
        caller,
        model,
        kind = "llm",
    ):
        state["unloads"].append(model)
        state["slots"]["llm"] = [m for m in state["slots"].get("llm", []) if m != model]

    monkeypatch.setattr(loading, "resident_slots", resident_slots)
    monkeypatch.setattr(loading, "unload_llm", unload_llm)
    monkeypatch.setattr(_common, "ensure_model", ensure_model)
    monkeypatch.delenv("UNSLOTH_FLOWS_PREFETCH", raising = False)
    return state


def _engine(handlers: dict) -> FlowEngine:
    eng = FlowEngine()
    eng.handlers = handlers
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    return eng


async def _run(eng: FlowEngine, d: dict, **options):
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = options)
    await asyncio.wait_for(rs.task, timeout = 30)
    return rs


async def _text(ctx):
    return {"text": str(ctx.params.get("value") or "")}


async def _gen(ctx):
    await ctx.ensure_model("chat", ctx.params.get("model"))
    return {"text": f"{ctx.params['model']['repo']}:{ctx.inputs.get('prompt')}"}


def _slow_template(spans: list):
    async def template(ctx):
        start = time.monotonic()
        await asyncio.sleep(0.6)
        spans.append((start, time.monotonic()))
        return {"text": "+".join(map(str, ctx.inputs.get("values") or []))}

    return template


def _chain(tag: str, **meta) -> dict:
    """Text -> LLM (m1) -> slow CPU template -> LLM (m2)."""
    return doc(
        [
            node("a", "input.text", value = tag),
            node("g1", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m1"}),
            node("t", "text.template", template = "{}"),
            node("g2", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m2"}),
        ],
        [
            edge("a", "text", "g1", "prompt"),
            edge("g1", "text", "t", "values"),
            edge("t", "text", "g2", "prompt"),
        ],
        **({"meta": meta} if meta else {}),
    )


@run_async
async def test_the_next_model_loads_while_cpu_steps_run(studio):
    spans: list = []
    eng = _engine(
        {"input.text": _text, "llm.generate": _gen, "text.template": _slow_template(spans)}
    )
    rs = await _run(eng, _chain("prefetch-on"))
    assert rs.status == "completed"
    assert [repo for repo, _ in studio["loads"]] == ["m1", "m2"]
    (start, end) = spans[0]
    m2_loaded = dict(studio["loads"])["m2"]
    assert start < m2_loaded < end
    assert rs.prefetched == 1
    logs = [
        e["data"]["message"] for e in rs.events if e["type"] == "node.log" and e["node_id"] == "g2"
    ]
    assert any("ahead while" in m for m in logs), logs


@run_async
async def test_the_kill_switch_keeps_loads_where_the_step_runs(studio, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FLOWS_PREFETCH", "0")
    spans: list = []
    eng = _engine(
        {"input.text": _text, "llm.generate": _gen, "text.template": _slow_template(spans)}
    )
    rs = await _run(eng, _chain("prefetch-off"))
    assert rs.status == "completed" and rs.prefetched == 0
    assert dict(studio["loads"])["m2"] > spans[0][1]


@run_async
async def test_a_rerun_served_from_the_cache_loads_nothing_ahead(studio):
    spans: list = []
    eng = _engine(
        {"input.text": _text, "llm.generate": _gen, "text.template": _slow_template(spans)}
    )
    await _run(eng, _chain("rerun"))
    studio["loads"].clear()
    studio["slots"]["llm"] = ["something-else"]
    rs = await _run(eng, _chain("rerun"))
    assert rs.nodes["g2"].status == "cached"
    ahead = [
        e for e in rs.events if e["type"] == "node.log" and "ahead while" in e["data"]["message"]
    ]
    assert ahead == [] and rs.prefetched == 0 and studio["loads"] == []


@run_async
async def test_nothing_loads_ahead_of_a_branch(studio):
    async def gate(ctx):
        await asyncio.sleep(0.4)
        return {"then": ctx.inputs.get("value")}

    eng = _engine({"input.text": _text, "llm.generate": _gen, "logic.if": gate})
    d = doc(
        [
            node("a", "input.text", value = "branch"),
            node("g1", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m1"}),
            node("if", "logic.if"),
            node("g2", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m2"}),
        ],
        [
            edge("a", "text", "g1", "prompt"),
            edge("g1", "text", "if", "value"),
            edge("if", "then", "g2", "prompt"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed" and rs.prefetched == 0


@run_async
async def test_the_first_gpu_step_keeps_what_studio_already_has_loaded(studio):
    studio["slots"]["llm"] = ["m2"]
    order = []

    async def gen(ctx):
        order.append(ctx.params["model"]["repo"])
        return await _gen(ctx)

    eng = _engine({"input.text": _text, "llm.generate": gen})
    d = doc(
        [
            node("g1", "llm.generate", prompt = "seeded one", model = {"repo": "m1"}),
            node("g2", "llm.generate", prompt = "seeded two", model = {"repo": "m2"}),
        ]
    )
    rs = await _run(eng, d)
    assert rs.status == "completed"
    assert order == ["m2", "m1"]
    assert [repo for repo, _ in studio["loads"]] == ["m1"]


@run_async
async def test_unload_after_the_run_frees_only_what_the_run_loaded(studio):
    studio["keep"] = True
    studio["slots"]["llm"] = ["m0", "m1"]
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    rs = await _run(eng, _chain("unload", unload_after_run = True))
    assert rs.status == "completed"
    assert studio["unloads"] == ["m2"]
    logs = [e["data"]["message"] for e in rs.events if e["type"] == "node.log"]
    assert any(m.startswith("Unloaded m2") for m in logs)


@run_async
async def test_models_stay_loaded_by_default(studio):
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    rs = await _run(eng, _chain("keep"))
    assert rs.status == "completed" and studio["unloads"] == []


def test_prefetch_switch_values(monkeypatch):
    for off in ("0", "false", "off"):
        monkeypatch.setenv("UNSLOTH_FLOWS_PREFETCH", off)
        assert not engine_mod.prefetch_enabled()
    monkeypatch.setenv("UNSLOTH_FLOWS_PREFETCH", "1")
    assert engine_mod.prefetch_enabled()


def test_the_plan_suggests_keeping_models_loaded_only_when_it_helps(monkeypatch):
    from routes import flows as flows_routes
    from utils import multi_model_settings

    two = [
        {"kind": "chat", "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF"},
        {"kind": "image", "repo": "black-forest-labs/FLUX.2-klein-4B"},
        {"kind": "embedding", "repo": "some/embedder"},
    ]
    monkeypatch.setattr(multi_model_settings, "get_multi_model_mode", lambda: "off")
    hints = flows_routes._residency_hints(two)
    assert [h["id"] for h in hints] == ["multi_model"] and "2 models" in hints[0]["message"]
    assert flows_routes._residency_hints(two[:1] + two[2:]) == []
    assert flows_routes._residency_hints([two[0], {**two[1], "optional": True}]) == []
    monkeypatch.setattr(multi_model_settings, "get_multi_model_mode", lambda: "fit")
    assert flows_routes._residency_hints(two) == []


@run_async
async def test_no_early_load_swaps_the_model_a_use_loaded_model_step_runs_on(studio):
    seen = {}

    async def gen(ctx):
        if not (ctx.params.get("model") or {}).get("repo"):
            seen[ctx.node_id] = list(studio["slots"].get("llm") or [])
            return {"text": "loaded"}
        return await _gen(ctx)

    async def template(ctx):
        await asyncio.sleep(0.5 if ctx.node_id == "t1" else 0.9)
        return {"text": "t"}

    eng = _engine({"input.text": _text, "llm.generate": gen, "text.template": template})
    d = doc(
        [
            node("a", "input.text", value = "loaded-model"),
            node("g1", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m1"}),
            node("t1", "text.template", template = "{}"),
            node("t2", "text.template", template = "{}"),
            node("loaded", "llm.generate", prompt = "{{prompt}}", model = {}),
            node("g2", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m2"}),
        ],
        [
            edge("a", "text", "g1", "prompt"),
            edge("g1", "text", "t1", "values"),
            edge("g1", "text", "t2", "values"),
            edge("t1", "text", "loaded", "prompt"),
            edge("t2", "text", "g2", "prompt"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed"
    assert seen["loaded"] == ["m1"]


@run_async
async def test_stopping_a_run_mid_early_load_stops_that_load(studio, monkeypatch):
    async def ensure_model(
        ctx,
        kind,
        choice,
        options = None,
    ):
        repo = choice["repo"]
        if repo == "m2":
            await asyncio.sleep(30)  # still loading when the run is stopped
        studio["slots"]["llm"] = [repo]
        ctx.resident["llm"] = repo
        return {"repo": repo}

    monkeypatch.setattr(_common, "ensure_model", ensure_model)
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    parsed, errors = parse_document(_chain("stop-early-load"))
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = parsed["id"], doc = parsed, options = {})
    rs.caller = rs.caller or object()
    for _ in range(100):
        if rs.prefetch_load:
            break
        await asyncio.sleep(0.05)
    assert rs.prefetch_load == ("llm", "m2")
    await eng.cancel(rs.id)
    await asyncio.wait_for(rs.task, timeout = 30)
    assert rs.status == "cancelled"
    assert studio["unloads"] == ["m2"]


@run_async
async def test_unload_after_the_run_leaves_a_slot_it_could_not_read(studio, monkeypatch):
    studio["keep"] = True
    studio["slots"]["llm"] = ["m1"]
    reads = []

    async def resident_slots(caller):
        reads.append(1)
        if len(reads) == 1:
            return {"image": []}
        return {k: list(v) for k, v in studio["slots"].items()}

    monkeypatch.setattr(loading, "resident_slots", resident_slots)
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    rs = await _run(eng, _chain("unknown-slot", unload_after_run = True))
    assert rs.status == "completed" and studio["unloads"] == []


@run_async
async def test_unload_after_the_run_waits_for_the_gpu(studio, monkeypatch):
    studio["slots"]["llm"] = []
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    d = _chain("unload-waits", unload_after_run = True)
    d["nodes"].append(node("tail", "text.template", template = "{}"))
    d["edges"].append(edge("g2", "text", "tail", "values"))
    parsed, errors = parse_document(d)
    assert not errors, errors
    lock = eng.gpu_lock()
    rs = await eng.start_run(app = None, flow_id = parsed["id"], doc = parsed, options = {})
    for _ in range(200):
        if rs.nodes["g2"].status in ("done", "cached"):
            break
        await asyncio.sleep(0.05)
    async with lock:
        await asyncio.sleep(1.5)
        assert studio["unloads"] == []
    await asyncio.wait_for(rs.task, timeout = 30)
    assert studio["unloads"] == ["m2"]


@run_async
async def test_unload_after_the_run_frees_an_agents_vision_model(studio):
    studio["keep"] = True
    studio["slots"]["llm"] = []

    async def agent(ctx):
        await ctx.ensure_model("vision", ctx.params.get("vision_model"))
        return {"text": "seen"}

    eng = _engine({"input.text": _text, "llm.generate": agent})
    d = doc(
        [
            node(
                "g",
                "llm.generate",
                prompt = "look",
                model = {"repo": "chat-m"},
                vision_model = {"repo": "vis-m"},
            )
        ],
        meta = {"unload_after_run": True},
    )
    rs = await _run(eng, d)
    assert rs.status == "completed" and studio["unloads"] == ["vis-m"]


@run_async
async def test_stopping_an_early_load_never_unloads_inside_another_flows_gpu_step(
    studio, monkeypatch
):
    async def ensure_model(
        ctx,
        kind,
        choice,
        options = None,
    ):
        if choice["repo"] == "m2":
            await asyncio.sleep(30)
        studio["slots"]["llm"] = [choice["repo"]]
        ctx.resident["llm"] = choice["repo"]
        return {"repo": choice["repo"]}

    inside = {"now": False, "seen": []}

    async def unload_llm(
        caller,
        model,
        kind = "llm",
    ):
        inside["seen"].append(inside["now"])

    monkeypatch.setattr(_common, "ensure_model", ensure_model)
    monkeypatch.setattr(loading, "unload_llm", unload_llm)
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    parsed, errors = parse_document(_chain("stop-early-load-locked"))
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = parsed["id"], doc = parsed, options = {})
    rs.caller = rs.caller or object()
    for _ in range(100):
        if rs.prefetch_load:
            break
        await asyncio.sleep(0.05)

    async def other_flow_step():
        async with eng.gpu_lock():
            inside["now"] = True
            await asyncio.sleep(0.5)
            inside["now"] = False

    other = asyncio.create_task(other_flow_step())
    await asyncio.sleep(0)
    await eng.cancel(rs.id)
    await asyncio.wait_for(asyncio.gather(rs.task, other), timeout = 30)
    assert inside["seen"] == [False]


@run_async
async def test_unload_after_the_run_leaves_a_model_another_run_loaded(studio, monkeypatch):
    reads = []

    async def resident_slots(caller):
        reads.append(1)
        return {"llm": []} if len(reads) == 1 else {k: list(v) for k, v in studio["slots"].items()}

    monkeypatch.setattr(loading, "resident_slots", resident_slots)
    studio["slots"]["llm"] = ["m1"]
    eng = _engine({"input.text": _text, "llm.generate": _gen})
    d = doc(
        [node("g", "llm.generate", prompt = "hi", model = {"repo": "m1"})],
        meta = {"unload_after_run": True},
    )
    rs = await _run(eng, d)
    assert rs.status == "completed" and studio["loads"] == [] and studio["unloads"] == []


@run_async
async def test_a_failed_early_load_leaves_the_steps_own_load_in_place(studio, monkeypatch):
    calls = []

    async def ensure_model(
        ctx,
        kind,
        choice,
        options = None,
    ):
        repo = choice["repo"]
        calls.append(repo)
        if repo == "m2" and calls.count("m2") == 1:
            raise RuntimeError("the early load failed")
        studio["slots"]["llm"] = [repo]
        ctx.resident["llm"] = repo
        return {"repo": repo}

    monkeypatch.setattr(_common, "ensure_model", ensure_model)
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    parsed, errors = parse_document(_chain("failed-early-load"))
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = parsed["id"], doc = parsed, options = {})
    rs.caller = rs.caller or object()
    await asyncio.wait_for(rs.task, timeout = 30)
    assert rs.status == "completed" and calls.count("m2") == 2
    assert studio["unloads"] == [] and studio["slots"]["llm"] == ["m2"]


@run_async
async def test_no_early_load_into_another_slot_evicts_the_model_a_use_loaded_model_step_needs(
    studio, monkeypatch
):
    seen = {}

    async def ensure_model(
        ctx,
        kind,
        choice,
        options = None,
    ):
        repo = (choice or {}).get("repo")
        if not repo:
            return None
        if kind == "image":
            await asyncio.sleep(0.2)
            studio["slots"]["llm"] = []
            studio["slots"]["image"] = [repo]
            ctx.resident["image"] = repo
            return {"repo": repo}
        studio["slots"]["llm"] = [repo]
        ctx.resident["llm"] = repo
        return {"repo": repo}

    async def gen(ctx):
        if not (ctx.params.get("model") or {}).get("repo"):
            seen[ctx.node_id] = list(studio["slots"].get("llm") or [])
            return {"text": "loaded"}
        return await _gen(ctx)

    async def image(ctx):
        await ctx.ensure_model("image", ctx.params.get("model"))
        return {"images": []}

    async def template(ctx):
        await asyncio.sleep(0.5 if ctx.node_id == "t1" else 0.9)
        return {"text": "t"}

    monkeypatch.setattr(_common, "ensure_model", ensure_model)
    eng = _engine(
        {
            "input.text": _text,
            "llm.generate": gen,
            "text.template": template,
            "image.generate": image,
        }
    )
    d = doc(
        [
            node("a", "input.text", value = "cross-slot"),
            node("g1", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m1"}),
            node("t1", "text.template", template = "{}"),
            node("t2", "text.template", template = "{}"),
            node("loaded", "llm.generate", prompt = "{{prompt}}", model = {}),
            node("img", "image.generate", model = {"repo": "img-m"}),
        ],
        [
            edge("a", "text", "g1", "prompt"),
            edge("g1", "text", "t1", "values"),
            edge("g1", "text", "t2", "values"),
            edge("t1", "text", "loaded", "prompt"),
            edge("t2", "text", "img", "prompt"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed", {k: v.error for k, v in rs.nodes.items()}
    assert seen["loaded"] == ["m1"]


@run_async
async def test_an_agent_seeing_an_image_on_the_loaded_vision_model_blocks_early_loads(studio):
    async def image(ctx):
        return {"image": {"kind": "image", "url": "/x.png"}}

    seen = {}

    async def agent(ctx):
        seen[ctx.node_id] = list(studio["slots"].get("llm") or [])
        return {"text": "seen"}

    async def template(ctx):
        await asyncio.sleep(0.5 if ctx.node_id == "t1" else 0.9)
        return {"text": "t"}

    eng = _engine(
        {
            "input.text": _text,
            "llm.generate": _gen,
            "text.template": template,
            "llm.agent": agent,
            "input.image": image,
        }
    )
    d = doc(
        [
            node("a", "input.text", value = "agent-vision"),
            node("pic", "input.image"),
            node("g1", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m1"}),
            node("t1", "text.template", template = "{}"),
            node("t2", "text.template", template = "{}"),
            node("ag", "llm.agent", model = {"repo": "chat-m"}),
            node("g2", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m2"}),
        ],
        [
            edge("a", "text", "g1", "prompt"),
            edge("g1", "text", "t1", "values"),
            edge("g1", "text", "t2", "values"),
            edge("t1", "text", "ag", "prompt"),
            edge("pic", "image", "ag", "image"),
            edge("t2", "text", "g2", "prompt"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed", {k: v.error for k, v in rs.nodes.items()}
    assert seen["ag"] == ["m1"]


@run_async
async def test_no_early_load_while_a_for_each_body_runs_on_the_loaded_model(studio):
    studio["slots"]["llm"] = ["m1"]
    seen = []

    async def items(ctx):
        return {"items": ["a", "b", "c"]}

    async def gen(ctx):
        if not (ctx.params.get("model") or {}).get("repo"):
            await asyncio.sleep(0.3)
            seen.append(list(studio["slots"].get("llm") or []))
            return {"text": "loaded"}
        return await _gen(ctx)

    eng = _engine({"input.list": items, "llm.generate": gen})
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each"),
            node("body", "llm.generate", prompt = "{{prompt}}", model = {}, group = "fe"),
            node("g2", "llm.generate", prompt = "sum up", model = {"repo": "m2"}),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "body", "prompt"),
            edge("body", "text", "fe", "collect"),
            edge("fe", "results", "g2", "context"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed", {k: v.error for k, v in rs.nodes.items()}
    assert seen == [["m1"]] * 3


@run_async
async def test_a_controlnet_image_step_is_not_loaded_early(studio):
    async def image(ctx):
        await ctx.ensure_model("image", ctx.params.get("model"))
        return {"images": []}

    async def pic(ctx):
        return {"image": {"kind": "image", "url": "/edges.png"}}

    eng = _engine(
        {
            "input.text": _text,
            "input.image": pic,
            "text.template": _slow_template([]),
            "image.generate": image,
        }
    )
    d = doc(
        [
            node("a", "input.text", value = "controlnet"),
            node("edges", "input.image"),
            node("t", "text.template", template = "{}"),
            node("img", "image.generate", model = {"repo": "img-m"}),
        ],
        [
            edge("a", "text", "t", "values"),
            edge("t", "text", "img", "prompt"),
            edge("edges", "image", "img", "control_image"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed", {k: v.error for k, v in rs.nodes.items()}
    assert rs.prefetched == 0


@run_async
async def test_a_player_invoke_that_ends_early_stops_its_early_load(studio, monkeypatch):
    async def ensure_model(
        ctx,
        kind,
        choice,
        options = None,
    ):
        if choice["repo"] == "m2":
            await asyncio.sleep(30)  # still loading when the invoke gives up
        return {"repo": choice["repo"]}

    async def failing(ctx):
        await asyncio.sleep(0.6)
        raise RuntimeError("upstream step failed")

    async def reply(ctx):
        return {"reply": ctx.inputs.get("reply")}

    monkeypatch.setattr(_common, "ensure_model", ensure_model)
    eng = _engine(
        {"input.text": _text, "llm.generate": _gen, "text.template": failing, "chat.output": reply}
    )
    d = doc(
        [
            node("a", "input.text", value = "invoke"),
            node("t", "text.template", template = "{}"),
            node("g2", "llm.generate", prompt = "{{prompt}}", model = {"repo": "m2"}),
            node("out", "chat.output"),
        ],
        [
            edge("a", "text", "t", "values"),
            edge("t", "text", "g2", "prompt"),
            edge("g2", "text", "out", "reply"),
        ],
    )
    parsed, errors = parse_document(d)
    assert not errors, errors
    with pytest.raises(Exception):
        await asyncio.wait_for(
            eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "out"), timeout = 20
        )
    assert studio["unloads"] == ["m2"]


@run_async
async def test_a_load_ahead_asks_for_the_cards_own_and_saved_load_settings(studio, monkeypatch):
    from core.flows import model_settings

    seen: dict = {}
    fake = _common.ensure_model

    async def ensure_model(
        ctx,
        kind,
        choice,
        options = None,
    ):
        seen.setdefault(choice["repo"], options)
        return await fake(ctx, kind, choice, options)

    monkeypatch.setattr(_common, "ensure_model", ensure_model)
    monkeypatch.setattr(
        model_settings, "saved_load", lambda repo, variant = None: {"kv_cache_dtype": "q8_0"}
    )
    d = _chain("prefetch-load")
    g2 = next(n for n in d["nodes"] if n["id"] == "g2")
    g2["params"]["load"] = {"max_seq_length": 8192}
    g2["params"]["use_model_settings"] = True
    eng = _engine({"input.text": _text, "llm.generate": _gen, "text.template": _slow_template([])})
    rs = await _run(eng, d)
    assert rs.status == "completed" and rs.prefetched == 1
    assert seen["m2"] == model_settings.card_load_kwargs(g2["params"])
    assert seen["m2"]["max_seq_length"] == 8192
    assert seen["m1"] is None
