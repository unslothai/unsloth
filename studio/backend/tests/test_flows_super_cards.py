# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
from dataclasses import asdict

from core.flows.compiler import compile_flow, parse_document
from ._flows_helpers import doc, edge, node


def _plan(d: dict) -> dict:
    parsed, errors = parse_document(d)
    assert not errors, errors
    plan = asdict(compile_flow(parsed))
    plan.pop("doc")
    return plan


def test_a_collapsed_super_card_compiles_to_the_same_plan():
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
    collapsed = copy.deepcopy(open_doc)
    collapsed["editor"] = {"super_cards": ["g"]}
    assert _plan(collapsed) == _plan(open_doc)
