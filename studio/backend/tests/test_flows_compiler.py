# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from core.flows import compiler
from core.flows.compiler import CompileError, compile_flow, parse_document
from core.flows.schema import MAX_NODES
from ._flows_helpers import doc, edge, node


def _compile(d: dict) -> compiler.Plan:
    parsed, errors = parse_document(d)
    assert not errors, errors
    return compile_flow(parsed)


def _issues(d: dict) -> list[dict]:
    parsed, errors = parse_document(d)
    if errors:
        return errors
    with pytest.raises(CompileError) as exc:
        compile_flow(parsed)
    return exc.value.issues


def test_topological_order_follows_data_edges_not_document_order():
    plan = _compile(
        doc(
            [
                node("out", "output.result"),
                node("t", "text.template"),
                node("a", "input.text", value = "x"),
            ],
            [edge("a", "text", "t", "values"), edge("t", "text", "out", "value")],
        )
    )
    assert plan.order == ["a", "t", "out"]


def test_type_mismatch_is_reported_on_the_edge():
    issues = _issues(
        doc(
            [node("img", "input.image"), node("split", "text.split")],
            [edge("img", "image", "split", "text")],
        )
    )
    assert issues[0]["edge_id"] == "img-image-split-text"
    assert "cannot connect" in issues[0]["message"]


def test_a_single_value_fits_a_list_port_of_its_type():
    plan = _compile(
        doc(
            [node("i", "input.image"), node("g", "image.grid")], [edge("i", "image", "g", "images")]
        )
    )
    assert plan.inputs["g"]["images"] == [("i", "image")]
    issues = _issues(
        doc([node("t", "input.text"), node("g", "image.grid")], [edge("t", "text", "g", "images")])
    )
    assert "cannot connect" in issues[0]["message"]


def test_json_ports_accept_text_and_lists_but_media_needs_media():
    plan = _compile(
        doc(
            [node("l", "input.list", items = ["a"]), node("g", "json.get", path = "0")],
            [edge("l", "items", "g", "json")],
        )
    )
    assert plan.order == ["l", "g"]


def test_cycle_is_refused():
    issues = _issues(
        doc(
            [node("a", "text.template"), node("b", "text.template")],
            [edge("a", "text", "b", "values"), edge("b", "text", "a", "values")],
        )
    )
    assert any("loop" in i["message"] for i in issues)


def test_single_input_takes_one_connection_multi_takes_many():
    issues = _issues(
        doc(
            [node("a", "input.text"), node("b", "input.text"), node("s", "text.split")],
            [edge("a", "text", "s", "text"), edge("b", "text", "s", "text")],
        )
    )
    assert any("takes one connection" in i["message"] for i in issues)
    plan = _compile(
        doc(
            [node("a", "input.text"), node("b", "input.text"), node("t", "text.template")],
            [edge("a", "text", "t", "values"), edge("b", "text", "t", "values")],
        )
    )
    assert plan.inputs["t"]["values"] == [("a", "text"), ("b", "text")]


def test_required_input_missing_but_param_fallback_counts():
    issues = _issues(doc([node("s", "text.split")]))
    assert issues[0]["node_id"] == "s"
    # llm.generate has an optional prompt port and a prompt param.
    _compile(doc([node("g", "llm.generate", prompt = "hi")]))


def test_unknown_type_and_bad_ports_are_errors():
    issues = _issues(doc([node("x", "made.up")]))
    assert "Unknown node type" in issues[0]["message"]
    issues = _issues(
        doc(
            [node("a", "input.text"), node("t", "text.template")],
            [edge("a", "nope", "t", "values")],
        )
    )
    assert "no output" in issues[0]["message"]


def test_flow_edges_only_between_story_handles():
    plan = _compile(
        doc(
            [node("s", "story.start"), node("sc", "story.scene"), node("e", "story.ending")],
            [edge("s", "next", "sc", "in", "flow"), edge("sc", "next", "e", "in", "flow")],
        )
    )
    # Player nodes are compiled but never executed.
    assert plan.order == []
    assert len(plan.flow_edges) == 2
    issues = _issues(
        doc(
            [node("a", "input.text"), node("sc", "story.scene")],
            [edge("a", "text", "sc", "in", "flow")],
        )
    )
    assert "no story exit" in issues[0]["message"]


def test_for_each_body_ordering_collect_and_escape_rules():
    nodes = [
        node("items", "input.list", items = ["a", "b"]),
        node("fe", "logic.for_each"),
        node("t", "text.template", group = "fe", template = "<{{values}}>"),
        node("out", "output.result"),
    ]
    edges = [
        edge("items", "items", "fe", "items"),
        edge("fe", "item", "t", "values"),
        edge("t", "text", "fe", "collect"),
        edge("fe", "results", "out", "value"),
    ]
    plan = _compile(doc(nodes, edges))
    assert plan.bodies == {"fe": ["t"]}
    assert plan.collect == {"fe": ("t", "text")}
    assert plan.top_level() == ["items", "fe", "out"]
    # A body node feeding the outside directly is refused.
    issues = _issues(doc(nodes, [*edges[:3], edge("t", "text", "out", "value")]))
    assert any("can only feed its group" in i["message"] for i in issues)


def test_for_each_waits_for_outside_inputs_of_its_body():
    nodes = [
        node("items", "input.list", items = ["a"]),
        node("fe", "logic.for_each"),
        node("t", "text.template", group = "fe"),
        node("late", "input.text", value = "z"),
    ]
    edges = [
        edge("items", "items", "fe", "items"),
        edge("fe", "item", "t", "values"),
        edge("late", "text", "t", "values"),
    ]
    plan = _compile(doc(nodes, edges))
    assert plan.order.index("late") < plan.order.index("fe")


def test_ancestors_closure_for_targets():
    plan = _compile(
        doc(
            [
                node("a", "input.text"),
                node("b", "input.text"),
                node("t", "text.template"),
                node("u", "text.template"),
            ],
            [edge("a", "text", "t", "values"), edge("b", "text", "u", "values")],
        )
    )
    assert plan.ancestors(["t"]) == {"a", "t"}


def test_limits_depth_and_format_guards():
    too_many = doc([node(f"n{i}", "input.text") for i in range(MAX_NODES + 1)])
    assert "Too many nodes" in parse_document(too_many)[1][0]["message"]
    deep: dict = {}
    cursor = deep
    for _ in range(100):
        cursor["x"] = {}
        cursor = cursor["x"]
    assert (
        "nested too deeply"
        in parse_document(doc([node("a", "input.text", value = deep)]))[1][0]["message"]
    )
    assert "format" in parse_document({**doc([]), "format": "comfy"})[1][0]["message"]
    assert "newer" in parse_document({**doc([]), "version": 99})[1][0]["message"]
    assert parse_document("{not json")[0] is None
    assert parse_document(json.dumps(doc([])))[0]["id"] == "f1"


def test_extra_fields_in_a_document_are_refused():
    parsed, errors = parse_document({**doc([]), "exec": "rm -rf /"})
    assert parsed is None and errors


def test_validate_endpoint_shape():
    assert compiler.validate(doc([node("a", "input.text")])) == {
        "ok": True,
        "errors": [],
        "warnings": [],
    }
    result = compiler.validate(doc([node("s", "text.split")]))
    assert result["ok"] is False and result["errors"][0]["node_id"] == "s"


def test_model_needs_merges_requirements_and_node_models():
    d = doc(
        [
            node(
                "g",
                "llm.generate",
                prompt = "x",
                model = {"repo": "unsloth/Qwen3-4B-GGUF", "variant": "Q4_K_M"},
            )
        ],
        requirements = {"models": [{"ref": "img", "kind": "image", "repo": "org/img"}]},
    )
    needs = compiler.model_needs(_compile(d))
    assert {(n["kind"], n["repo"]) for n in needs} == {
        ("image", "org/img"),
        ("chat", "unsloth/Qwen3-4B-GGUF"),
    }
