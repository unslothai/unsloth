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


def test_an_exact_id_wins_over_another_blocks_label():
    d = doc(
        [
            {**node("a", "input.text", value = "x"), "label": "b"},
            node("b", "input.text", value = "y"),
        ]
    )
    filled, _ = apply_input_values(d, {"b": "set"})
    by_id = {n["id"]: n for n in filled["nodes"]}
    assert by_id["b"]["params"]["value"] == "set"
    assert by_id["a"]["params"]["value"] == "x"


@run_async
async def test_a_pin_wins_over_a_baked_output():
    eng = _engine(BASE)
    d = doc(
        [node("a", "input.text", value = "x"), node("out", "output.result", name = "r")],
        [edge("a", "text", "out", "value")],
    )
    d["baked"] = {"a": {"text": "baked"}}
    rs = await _run(eng, d, pinned = {"a": {"text": "pinned!"}})
    assert rs.outputs["r"] == "pinned!"


def test_history_finds_distinct_outputs_behind_many_repeats():
    for i in range(3):
        flows_db.create_run(f"old{i}", "hist", {})
        flows_db.save_node_result(f"old{i}", "a", "done", {"text": f"old{i}"}, None)
    conn = flows_db.get_connection()
    with conn:
        conn.executemany(
            "INSERT INTO flow_runs (id, flow_id, status, options_json, created_at) VALUES (?, 'hist', 'completed', '{}', ?)",
            [(f"rep{i}", 10**15 + i) for i in range(600)],
        )
        conn.executemany(
            "INSERT INTO flow_node_results (run_id, node_id, status, outputs_json, updated_at) VALUES (?, 'a', 'cached', ?, 0)",
            [(f"rep{i}", '{"text": "same"}') for i in range(600)],
        )
    conn.close()
    texts = [e["outputs"]["text"] for e in flows_db.node_output_history("hist", "a")]
    assert texts[0] == "same" and sorted(texts[1:]) == ["old0", "old1", "old2"]


@run_async
async def test_called_flows_share_the_callers_gpu_step_limit(monkeypatch):
    from core.flows import engine as engine_mod
    from core.flows.specs import SPECS_BY_TYPE, out, spec

    async def gpu_text(ctx):
        return {"text": "g"}

    monkeypatch.setitem(
        SPECS_BY_TYPE,
        "gpu.text",
        spec(
            "gpu.text",
            "GPU",
            "text",
            "test",
            outputs = [out("text", "text")],
            lane = "gpu",
            cacheable = False,
        ),
    )
    d = doc(
        [node("g", "gpu.text"), node("out", "output.result", name = "answer")],
        [edge("g", "text", "out", "value")],
    )
    d["id"] = "gpuchild"
    flows_db.save_flow(parse_document(d)[0])
    parent = doc(
        [
            node("c1", "flow.call", flow_id = "gpuchild", values = {}),
            node("c2", "flow.call", flow_id = "gpuchild", values = {}),
            node("o1", "output.result", name = "one"),
            node("o2", "output.result", name = "two"),
        ],
        [edge("c1", "outputs", "o1", "value"), edge("c2", "outputs", "o2", "value")],
        id = "parent",
    )
    eng = _engine(_handlers(**{"gpu.text": gpu_text}))
    rs = await _run(eng, parent)
    assert rs.status == "completed", rs.error
    monkeypatch.setattr(engine_mod, "MAX_GPU_OPS_PER_RUN", 1)
    rs = await _run(eng, parent)
    assert rs.status == "failed"


def test_declared_flow_params_are_passed_through():
    d = {"nodes": [], "params": [{"id": "topic", "type": "text"}]}
    _, params = apply_input_values(d, {"topic": "otters", "unknown": 1})
    assert params == {"topic": "otters"}


@run_async
async def test_run_flow_keeps_a_declared_params_default():
    d = doc(
        [
            node("mood", "input.param", name = "mood", type = "text"),
            node("t", "text.template", template = "feeling {}"),
            node("out", "output.result", name = "answer"),
        ],
        [edge("mood", "value", "t", "values"), edge("t", "text", "out", "value")],
        params = [{"id": "mood", "type": "text", "default": "calm"}],
    )
    d["id"] = "moody"
    parsed, errors = parse_document(d)
    assert not errors, errors
    flows_db.save_flow(parsed)
    eng = _engine(_handlers())
    rs = await _run(eng, _caller_doc("moody", {}))
    assert rs.status == "completed", rs.error
    assert rs.outputs["got"] == {"answer": "feeling calm"}


def test_a_pin_holding_deleted_media_runs_the_block_again(monkeypatch):
    from core.flows.nodes import _common

    calls = []

    async def counted(ctx):
        calls.append(1)
        return {"text": "fresh"}

    eng = _engine({**BASE, "input.text": counted})
    d = doc(
        [node("a", "input.text", value = "x"), node("out", "output.result", name = "r")],
        [edge("a", "text", "out", "value")],
    )
    pin = {"a": {"text": {"ref": "image:gone", "kind": "image"}}}
    monkeypatch.setattr(_common, "gallery_media_present", lambda value: False)
    rs = asyncio.run(_run(eng, d, pinned = pin))
    assert rs.outputs["r"] == "fresh" and calls == [1]
    monkeypatch.setattr(_common, "gallery_media_present", lambda value: True)
    rs = asyncio.run(_run(eng, d, pinned = pin))
    assert rs.nodes["a"].status == "cached" and calls == [1]


@run_async
async def test_rerolling_a_run_flow_block_reruns_the_called_flow():
    _child("rechild")
    calls = []

    async def template(ctx):
        calls.append(1)
        return {"text": f"take {len(calls)}"}

    eng = _engine(_handlers(**{"text.template": template}))
    d = _caller_doc("rechild", {"topic": "otters"})
    first = await _run(eng, d)
    again = await _run(eng, d)
    assert again.outputs["got"] == first.outputs["got"] and len(calls) == 1
    rerolled = await _run(eng, d, force_nodes = ["call"])
    assert len(calls) == 2 and rerolled.outputs["got"] != first.outputs["got"]


def test_a_called_flows_document_input_takes_a_value():
    d = doc([{**node("f", "input.file"), "label": "Report"}])
    media = {"kind": "document", "ref": "sha256:" + "a" * 64, "name": "q3.pdf"}
    filled, _ = apply_input_values(d, {"report": media})
    assert filled["nodes"][0]["params"]["asset"] == media
