# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The run supervisor with fake handlers: ordering, skips and joins, For Each, cache, cancel, failure isolation,
GPU swap ordering and event replay."""

import asyncio

import pytest

from core.flows import engine as engine_mod
from core.flows.compiler import parse_document
from core.flows.engine import FlowEngine
from core.flows.registry import FlowNodeError
from storage import flows_db

from ._flows_helpers import doc, edge, node, run_async


def _engine(handlers: dict) -> FlowEngine:
    eng = FlowEngine()
    eng.handlers = handlers
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    return eng


async def _text(ctx):
    return {"text": str(ctx.params.get("value") or "")}


async def _template(ctx):
    values = ctx.inputs.get("values") or []
    return {"text": (ctx.params.get("template") or "{}").replace("{}", "+".join(map(str, values)))}


async def _result(ctx):
    return {"value": ctx.inputs.get("value")}


BASE = {
    "input.text": _text,
    "text.template": _template,
    "output.result": _result,
}


async def _run(eng: FlowEngine, d: dict, **options):
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = options)
    await asyncio.wait_for(rs.task, timeout = 20)
    return rs


def _status(rs, nid):
    return rs.nodes[nid].status


@run_async
async def test_linear_run_produces_outputs_and_ordered_events():
    eng = _engine(dict(BASE))
    d = doc(
        [
            node("a", "input.text", value = "hi"),
            node("t", "text.template", template = "<{}>"),
            node("o", "output.result", name = "answer"),
        ],
        [edge("a", "text", "t", "values"), edge("t", "text", "o", "value")],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed"
    assert rs.outputs == {"answer": "<hi>"}
    seqs = [e["seq"] for e in rs.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    stored = flows_db.events_after(rs.id, 0)
    assert [e["seq"] for e in stored] == seqs
    assert stored[-1]["type"] == "run.completed"
    assert flows_db.get_run(rs.id)["status"] == "completed"
    assert flows_db.node_results(rs.id)["t"]["outputs"] == {"text": "<hi>"}
    # Replay from the middle.
    assert [e["seq"] for e in flows_db.events_after(rs.id, seqs[3])] == seqs[4:]


@run_async
async def test_if_skips_the_branch_not_taken_and_merge_joins():
    calls = []

    async def branch(ctx):
        return (
            {"then": ctx.inputs["value"]}
            if ctx.inputs["value"] == "yes"
            else {"else": ctx.inputs["value"]}
        )

    async def tagged(ctx):
        calls.append(ctx.node_id)
        return {"text": f"{ctx.node_id}:{ctx.inputs['values'][0]}"}

    async def merge(ctx):
        for port in ("a", "b", "c"):
            if port in ctx.inputs:
                return {"value": ctx.inputs[port]}
        return {}

    eng = _engine({**BASE, "logic.if": branch, "text.template": tagged, "logic.merge": merge})
    d = doc(
        [
            node("a", "input.text", value = "no"),
            node("if", "logic.if"),
            node("yes", "text.template"),
            node("no", "text.template"),
            node("m", "logic.merge"),
            node("o", "output.result", name = "r"),
        ],
        [
            edge("a", "text", "if", "value"),
            edge("if", "then", "yes", "values"),
            edge("if", "else", "no", "values"),
            edge("yes", "text", "m", "a"),
            edge("no", "text", "m", "b"),
            edge("m", "value", "o", "value"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed"
    assert _status(rs, "yes") == "skipped"
    assert calls == ["no"]
    assert rs.outputs == {"r": "no:no"}


@run_async
async def test_failure_is_isolated_and_downstream_skipped():
    async def boom(ctx):
        raise FlowNodeError("nope")

    eng = _engine({**BASE, "text.split": boom})
    d = doc(
        [
            node("a", "input.text", value = "x"),
            node("bad", "text.split"),
            node("after", "list.join"),
            node("ok", "text.template", template = "[{}]"),
            node("o", "output.result", name = "ok"),
        ],
        [
            edge("a", "text", "bad", "text"),
            edge("bad", "items", "after", "items"),
            edge("a", "text", "ok", "values"),
            edge("ok", "text", "o", "value"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "failed"
    assert rs.nodes["bad"].error == "nope"
    assert _status(rs, "after") == "skipped"
    assert rs.nodes["after"].message == "An upstream step failed"
    assert rs.outputs == {"ok": "[x]"}


@run_async
async def test_for_each_runs_body_per_item_and_collects():
    eng = _engine(
        {
            **BASE,
            "input.list": lambda ctx: _ret({"items": ["a", "b", "c"]}),
            "text.template": lambda ctx: _ret({"text": str(ctx.inputs["values"][0]).upper()}),
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
            node("o", "output.result", name = "all"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
            edge("fe", "results", "o", "value"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed", rs.error
    assert rs.outputs == {"all": ["A", "B", "C"]}
    iterations = [
        e["data"].get("iteration")
        for e in rs.events
        if e["type"] == "node.output" and e["node_id"] == "t"
    ]
    assert iterations == [[0], [1], [2]]


@run_async
async def test_for_each_waits_for_outside_inputs_of_its_body():
    async def slow_ref(ctx):
        await asyncio.sleep(0.3)
        return {"text": "ref"}

    seen = []

    async def body(ctx):
        seen.append(ctx.inputs.get("values"))
        return {"text": "x"}

    eng = _engine(
        {
            **BASE,
            "llm.generate": slow_ref,
            "input.list": lambda ctx: _ret({"items": ["a", "b"]}),
            "text.template": body,
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("ref", "llm.generate"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("ref", "text", "t", "values"),
            edge("t", "text", "fe", "collect"),
        ],
    )
    rs = await _run(eng, d)
    assert rs.status == "completed", rs.error
    assert _status(rs, "t") != "skipped" and len(seen) == 2 and all("ref" in v for v in seen)


async def _ret(value):
    return value


@run_async
async def test_cache_hits_on_second_run_and_force_bypasses():
    calls = []

    async def counted(ctx):
        calls.append(ctx.node_id)
        return {"text": "v"}

    eng = _engine({**BASE, "text.template": counted})
    d = doc(
        [node("a", "input.text", value = "k"), node("t", "text.template")],
        [edge("a", "text", "t", "values")],
    )
    first = await _run(eng, d)
    second = await _run(eng, d)
    assert calls == ["t"]
    assert _status(first, "t") == "done" and _status(second, "t") == "cached"
    # Moving a node does not change its key.
    moved = {**d, "nodes": [{**n, "pos": [500, 500]} for n in d["nodes"]]}
    assert _status(await _run(eng, moved), "t") == "cached"
    await _run(eng, d, force_nodes = ["t"])
    assert calls == ["t", "t"]
    # A changed param is a new key.
    changed = {**d, "nodes": [d["nodes"][0], {**d["nodes"][1], "params": {"template": "other"}}]}
    await _run(eng, changed)
    assert calls == ["t", "t", "t"]


@run_async
async def test_side_effect_nodes_never_cache():
    calls = []

    async def save(ctx):
        calls.append(1)
        return {"saved": []}

    eng = _engine({**BASE, "output.library": save})
    d = doc(
        [node("a", "input.text", value = "k"), node("s", "output.library")],
        [edge("a", "text", "s", "value")],
    )
    await _run(eng, d)
    await _run(eng, d)
    assert calls == [1, 1]


@run_async
async def test_cancel_mid_run_keeps_finished_outputs_and_fires_hooks():
    started = asyncio.Event()
    hooked = []

    async def slow(ctx):
        async def hook():
            hooked.append(ctx.node_id)

        ctx.on_cancel(hook)
        started.set()
        for _ in range(200):
            await asyncio.sleep(0.02)
            ctx.check_cancelled()
        return {"text": "late"}

    eng = _engine({**BASE, "llm.generate": slow})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate"), node("t", "text.template")],
        [edge("a", "text", "g", "prompt"), edge("g", "text", "t", "values")],
    )
    parsed, _ = parse_document(d)
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
    await asyncio.wait_for(started.wait(), 60)
    assert await eng.cancel(rs.id) == "cancelling"
    await asyncio.wait_for(rs.task, 60)
    assert rs.status == "cancelled"
    assert hooked == ["g"]
    assert _status(rs, "a") == "done"
    assert _status(rs, "g") == "cancelled"
    assert _status(rs, "t") in ("cancelled", "skipped")
    assert flows_db.get_run(rs.id)["status"] == "cancelled"
    assert rs.events[-1]["type"] == "run.cancelled"


@run_async
async def test_gpu_nodes_prefer_the_resident_model():
    order = []

    async def gen(ctx):
        repo = ctx.params["model"]["repo"]
        order.append(repo)
        ctx.resident["llm"] = repo
        return {"text": repo}

    eng = _engine({**BASE, "llm.generate": gen})
    models = ["m1", "m2", "m1", "m2", "m1"]
    nodes = [
        node(f"g{i}", "llm.generate", prompt = "x", model = {"repo": m}) for i, m in enumerate(models)
    ]
    rs = await _run(eng, doc(nodes))
    assert rs.status == "completed"
    assert order == ["m1", "m1", "m1", "m2", "m2"]


@run_async
async def test_target_nodes_run_only_their_ancestors():
    calls = []

    async def counted(ctx):
        calls.append(ctx.node_id)
        return {"text": ctx.node_id}

    eng = _engine({"input.text": counted, "text.template": counted})
    d = doc(
        [node("a", "input.text"), node("b", "input.text"), node("t", "text.template")],
        [edge("a", "text", "t", "values")],
    )
    await _run(eng, d, target_nodes = ["t"])
    assert sorted(calls) == ["a", "t"]


@run_async
async def test_baked_outputs_are_used_without_running():
    async def never(ctx):
        raise AssertionError("baked node ran")

    eng = _engine({**BASE, "llm.generate": never})
    d = doc(
        [node("g", "llm.generate", prompt = "x"), node("o", "output.result", name = "r")],
        [edge("g", "text", "o", "value")],
        baked = {"g": {"text": "from the package"}},
    )
    rs = await _run(eng, d)
    assert rs.outputs == {"r": "from the package"}
    assert rs.nodes["g"].message == "Baked"


@run_async
async def test_invoke_runs_entrypoint_with_supplied_inputs():
    async def next_scene(ctx):
        return {"scene": {"state": ctx.inputs.get("state"), "seed": ctx.seed}}

    eng = _engine({**BASE, "story.next_scene": next_scene})
    d = doc([node("n", "story.next_scene")])
    parsed, _ = parse_document(d)
    out = await eng.invoke(
        app = None, flow_id = "f1", doc = parsed, entry = "n", inputs = {"state": {"hp": 3}}
    )
    assert out["scene"]["state"] == {"hp": 3}
    with pytest.raises(FlowNodeError):
        await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "missing")
    d2 = doc([node("a", "input.text")])
    parsed2, _ = parse_document(d2)
    with pytest.raises(FlowNodeError, match = "not an entry point"):
        await eng.invoke(app = None, flow_id = "f1", doc = parsed2, entry = "a")


@run_async
async def test_run_is_invisible_to_another_account():
    from utils.account_context import AccountContext, arun_as

    eng = _engine(dict(BASE))
    rs = await _run(eng, doc([node("a", "input.text", value = "x")]))
    bob = AccountContext("bob-flows", "bob")
    assert await arun_as(bob, _live(eng, rs.id)) is None


async def _live(eng, run_id):
    return eng.live(run_id)


def test_orphaned_runs_are_interrupted():
    flows_db.create_run("orphan", "f1", {})
    flows_db.update_run("orphan", status = "running")
    assert flows_db.reconcile_orphaned_runs() >= 1
    assert flows_db.get_run("orphan")["status"] == "interrupted"
    # A terminal status is never reopened.
    flows_db.update_run("orphan", status = "running")
    assert flows_db.get_run("orphan")["status"] == "interrupted"


def test_progress_is_coalesced(monkeypatch):
    from core.flows.engine import RunState

    times = iter([0.0, 0.05, 0.1, 0.3])
    monkeypatch.setattr(engine_mod.time, "monotonic", lambda: next(times))
    rs = RunState(id = "r", flow_id = "f", account = None, doc = {}, plan = None, options = {}, persist = False)
    for i in range(3):
        rs.progress("n", {"fraction": i / 10})
    assert [e["data"]["fraction"] for e in rs.events] == [0.0]
    rs.flush()
    assert [e["data"]["fraction"] for e in rs.events] == [0.0, 0.2]
