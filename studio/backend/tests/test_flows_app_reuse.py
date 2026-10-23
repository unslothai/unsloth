# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""App inputs, Run Flow (flow.call) and per-node output history with pins."""

import asyncio

from core.flows.compiler import parse_document
from core.flows.nodes import logic
from core.flows.nodes.inputs import apply_input_values, param
from storage import flows_db

from ._flows_helpers import doc, edge, node, run_async
from .test_flows_engine import BASE, _engine, _run, _ret


def _child(flow_id: str, **extra) -> dict:
    d = doc(
        [
            node("topic", "input.param", name = "topic", type = "text", default = "cats"),
            node("t", "text.template", template = "about {}"),
            node("out", "output.result", name = "answer"),
        ],
        [edge("topic", "value", "t", "values"), edge("t", "text", "out", "value")],
        **extra,
    )
    d["id"] = flow_id
    parsed, errors = parse_document(d)
    assert not errors, errors
    flows_db.save_flow(parsed)
    return parsed


def _caller_doc(flow_id: str, values: dict) -> dict:
    return doc(
        [
            node("call", "flow.call", flow_id = flow_id, values = values),
            node("out", "output.result", name = "got"),
        ],
        [edge("call", "outputs", "out", "value")],
        id = "parent",
    )


def _handlers(**extra):
    return {**BASE, "input.param": param, "flow.call": logic.call_flow, **extra}


def test_input_values_fill_flow_inputs_and_blocks_by_label():
    d = doc(
        [
            node("p", "input.param", name = "topic", type = "text"),
            {**node("img", "input.image"), "label": "Photo"},
            node("n", "input.number", value = 1),
            node("s", "input.seed", value = 1, randomize = True),
        ]
    )
    media = {"kind": "image", "ref": "image:1"}
    filled, params = apply_input_values(
        d, {"topic": "otters", "photo": media, "n": 7, "s": 42, "zzz": 1}
    )
    by_id = {n["id"]: n for n in filled["nodes"]}
    assert params == {"topic": "otters"}
    assert by_id["img"]["params"]["asset"] == media
    assert by_id["n"]["params"]["value"] == 7
    assert by_id["s"]["params"] == {"value": 42, "randomize": False}
    # The stored flow is never touched.
    assert d["nodes"][1]["params"] == {}


@run_async
async def test_run_flow_returns_the_called_flows_outputs():
    _child("child")
    eng = _engine(_handlers())
    rs = await _run(eng, _caller_doc("child", {"topic": "otters"}))
    assert rs.status == "completed", rs.error
    assert rs.outputs["got"] == {"answer": "about otters"}


@run_async
async def test_run_flow_refuses_cycles_and_deep_chains():
    # a -> b -> a is a cycle.
    for fid, target in (("a", "b"), ("b", "a")):
        d = _caller_doc(target, {})
        d["id"] = fid
        flows_db.save_flow(parse_document(d)[0])
    eng = _engine(_handlers())
    rs = await _run(eng, {**_caller_doc("b", {}), "id": "a"})
    assert rs.status == "failed"
    assert "already running" in (rs.nodes["call"].error or "")

    # d1 -> d2 -> d3 -> d4 -> leaf: the fourth hop is refused.
    _child("leaf")
    chain = ["d1", "d2", "d3", "d4"]
    for fid, target in zip(chain, chain[1:] + ["leaf"]):
        d = _caller_doc(target, {})
        d["id"] = fid
        flows_db.save_flow(parse_document(d)[0])
    rs = await _run(eng, {**_caller_doc("d2", {}), "id": "d1"})
    assert rs.status == "failed"
    assert "at most 3" in (rs.nodes["call"].error or "")


@run_async
async def test_run_flow_names_a_missing_flow():
    eng = _engine(_handlers())
    rs = await _run(eng, _caller_doc("nope", {}))
    assert rs.status == "failed"
    assert "not in your Flows" in (rs.nodes["call"].error or "")


@run_async
async def test_stopping_the_caller_stops_the_called_flow(monkeypatch):
    stopped = []

    async def slow(ctx):
        async def hook():
            stopped.append(ctx.node_id)

        ctx.on_cancel(hook)
        for _ in range(200):
            await asyncio.sleep(0.05)
            ctx.check_cancelled()
        return {"text": "late"}

    from core.flows.specs import SPECS_BY_TYPE, out, spec

    monkeypatch.setitem(
        SPECS_BY_TYPE,
        "slow.text",
        spec("slow.text", "Slow", "text", "test", outputs = [out("text", "text")]),
    )
    d = doc(
        [node("t", "slow.text"), node("out", "output.result", name = "answer")],
        [edge("t", "text", "out", "value")],
    )
    d["id"] = "slowchild"
    flows_db.save_flow(parse_document(d)[0])
    eng = _engine(_handlers(**{"slow.text": slow}))
    parsed, _ = parse_document(_caller_doc("slowchild", {}))
    rs = await eng.start_run(app = None, flow_id = "parent", doc = parsed, options = {})
    for _ in range(100):
        await asyncio.sleep(0.05)
        if stopped or rs.nodes["call"].status == "running":
            break
    await asyncio.sleep(0.2)
    await eng.cancel(rs.id)
    await asyncio.wait_for(rs.task, timeout = 20)
    assert rs.status == "cancelled"
    assert stopped == ["t"]


@run_async
async def test_history_keeps_distinct_outputs_and_pins_feed_downstream():
    calls = []

    async def counted(ctx):
        calls.append(1)
        return {"text": f"v{len(calls)}"}

    eng = _engine({**BASE, "input.text": counted})
    d = doc(
        [node("a", "input.text", value = "x"), node("out", "output.result", name = "r")],
        [edge("a", "text", "out", "value")],
    )
    for _ in range(12):
        await _run(eng, d, force_nodes = ["a"])
    entries = flows_db.node_output_history("f1", "a")
    assert len(entries) == 10
    assert entries[0]["outputs"] == {"text": "v12"} and entries[-1]["outputs"] == {"text": "v3"}

    rs = await _run(eng, d, pinned = {"a": {"text": "pinned!"}})
    assert rs.outputs["r"] == "pinned!" and rs.nodes["a"].status == "cached"
    assert len(calls) == 12
    # Regenerate (force) ignores the pin.
    rs = await _run(eng, d, force_nodes = ["a"], pinned = {"a": {"text": "pinned!"}})
    assert rs.outputs["r"] == "v13"


def test_run_flow_is_in_the_catalog_and_never_cached():
    from core.flows.specs import SPECS_BY_TYPE

    s = SPECS_BY_TYPE["flow.call"]
    assert s["lane"] == "cpu" and not s["cacheable"]
    assert _ret  # helper import kept for parity with the engine tests
