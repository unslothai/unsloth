# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model residency across a run: start from what Studio has loaded, load the next step's model while CPU steps
run, never load ahead for a step that will be a cache hit, and free what the run loaded when asked to."""

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
    """Fake Studio residency: what each slot holds, every load the run makes, and every unload."""
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
            # "keep": Keep multiple models loaded, so a load adds to what is resident instead of replacing it.
            state["slots"]["llm"] = [*held, repo] if state.get("keep") else [repo]
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
    # m2 was ready before the template that feeds it finished, so the second LLM step did not wait for a load.
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
    # Nothing upstream computed fresh output, so every step was a cache hit: no early load may even start.
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
    # m2 was already resident, so it runs first and m1 loads once, instead of m2 being evicted and reloaded.
    assert order == ["m2", "m1"]
    assert [repo for repo, _ in studio["loads"]] == ["m1"]


@run_async
async def test_unload_after_the_run_frees_only_what_the_run_loaded(studio):
    # m1 was loaded before the run (and the run uses it), m0 too (unrelated): only m2, which the run loaded, goes.
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
    # One model (an embedder does not take turns), an optional second, or a setting that already keeps them: no hint.
    assert flows_routes._residency_hints(two[:1] + two[2:]) == []
    assert flows_routes._residency_hints([two[0], {**two[1], "optional": True}]) == []
    monkeypatch.setattr(multi_model_settings, "get_multi_model_mode", lambda: "fit")
    assert flows_routes._residency_hints(two) == []
