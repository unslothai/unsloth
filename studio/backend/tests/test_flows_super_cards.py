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


def test_a_super_card_compiles_to_the_same_plan():
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
    for cards in (
        [{"id": "sc1", "title": "Wrap", "members": ["t", "t2"]}],
        [{"id": "sc1", "title": "Wrap", "members": ["t", "t2"], "open": True}],
        ["g"],  # the earlier group-id format
    ):
        combined = copy.deepcopy(open_doc)
        combined["editor"] = {"super_cards": cards}
        assert _plan(combined) == _plan(open_doc)
