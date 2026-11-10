# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import json
from dataclasses import asdict

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


def test_for_each_groups_owning_each_other_are_refused_without_hanging():
    import threading

    nodes = [
        node("items", "input.list", items = ["a"]),
        node("a", "logic.for_each", group = "b"),
        node("b", "logic.for_each", group = "a"),
        node("t", "text.template", group = "a"),
    ]
    edges = [
        edge("items", "items", "a", "items"),
        edge("a", "item", "t", "values"),
        edge("items", "items", "b", "items"),
    ]
    found: list = []
    worker = threading.Thread(target = lambda: found.append(_issues(doc(nodes, edges))), daemon = True)
    worker.start()
    worker.join(10)
    assert found, "compiling an ownership cycle hung"
    assert any("nested in a loop" in i["message"] for i in found[0])


def test_two_flow_outputs_cannot_share_a_name():
    d = doc([node("o1", "output.result", name = "x"), node("o2", "output.result", name = "x")])
    parsed, _ = parse_document(d)
    with pytest.raises(CompileError) as err:
        compile_flow(parsed)
    assert any("already named" in i["message"] for i in err.value.issues)


def _plan(d: dict) -> dict:
    parsed, errors = parse_document(d)
    assert not errors, errors
    plan = asdict(compile_flow(parsed))
    plan.pop("doc")
    return plan


@pytest.mark.parametrize(
    "cards",
    [
        [{"id": "sc1", "title": "Wrap", "members": ["t", "t2"]}],
        [{"id": "sc1", "title": "Wrap", "members": ["t", "t2"], "open": True}],
        ["g"],  # the earlier group-id format
    ],
    ids = ["closed", "open", "group-id"],
)
def test_a_super_card_compiles_to_the_same_plan(cards):
    open_doc = doc(
        [
            node("a", "input.text", value = "x"),
            node("t", "text.template", group = "g", template = "<{{values}}>"),
            node("t2", "text.template", group = "g", template = "[{{values}}]"),
            node("out", "output.result"),
        ],
        [
            edge("a", "text", "t", "values"),
            edge("t", "text", "t2", "values"),
            edge("t2", "text", "out", "value"),
        ],
        groups = [{"id": "g", "label": "Wrap", "rect": [0, 0, 400, 300]}],
    )
    combined = copy.deepcopy(open_doc)
    combined["editor"] = {"super_cards": cards}
    assert _plan(combined) == _plan(open_doc)


def test_upstream_sets_are_worked_out_once_per_node(monkeypatch):
    from core.flows import compiler

    calls = []
    real = compiler.Plan.ancestors
    monkeypatch.setattr(
        compiler.Plan, "ancestors", lambda self, t: calls.append(1) or real(self, t)
    )
    nodes = [node(f"t{i}", "text.template") for i in range(30)]
    edges = [edge(f"t{i}", "text", f"t{i + 1}", "values") for i in range(29)]
    plan = compiler.compile_flow(parse_document(doc(nodes, edges))[0])
    calls.clear()
    for _ in range(20):
        for i in range(30):
            plan.upstream(f"t{i}")
    assert len(calls) == 30
    assert plan.upstream("t3") == {"t0", "t1", "t2"}


def test_the_heap_topological_order_matches_the_sorted_list_one():
    import random

    from core.flows.compiler import _topo

    def reference(nodes, inputs):
        position = {nid: i for i, nid in enumerate(nodes)}
        remaining = {n: {s for ss in inputs.get(n, {}).values() for s, _ in ss} for n in nodes}
        dependents = {n: set() for n in nodes}
        for n, d in remaining.items():
            for s in d:
                dependents[s].add(n)
        ready = sorted((n for n, d in remaining.items() if not d), key = position.__getitem__)
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for nxt in dependents[n]:
                remaining[nxt].discard(n)
                if not remaining[nxt] and nxt not in order and nxt not in ready:
                    ready.append(nxt)
            ready.sort(key = position.__getitem__)
        return order if len(order) == len(nodes) else None

    rng = random.Random(0)
    for _ in range(200):
        names = [f"n{i}" for i in range(rng.randint(1, 25))]
        rng.shuffle(names)
        nodes = {n: {} for n in names}
        inputs = {}
        for i, n in enumerate(names):
            pool = names if rng.random() < 0.05 else names[:i] + names[i + 1 :]
            picks = rng.sample(pool, min(len(pool), rng.randint(0, 3)))
            if picks:
                inputs[n] = {"a": [(p, "o") for p in picks]}
        assert _topo(nodes, inputs, {n: None for n in names}, {}) == reference(nodes, inputs)
