# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from eval.json_score import json_anls_score, score_from_text
from eval.json_score.core import ScoreNode


SCHEMA = {
    "total": {"type": "money", "rel_tol": 0.0},
    "currency": "categorical",
    "issue_date": {"type": "date"},
    "line_items": [{"desc": "string", "qty": "numeric", "price": "money"}],
}


def _gt():
    return {
        "total": 130,
        "currency": "USD",
        "issue_date": "2024-01-15",
        "line_items": [
            {"desc": "Apple", "qty": 1, "price": 10},
            {"desc": "Banana", "qty": 2, "price": 60},
        ],
    }


def test_perfect_score():
    assert json_anls_score(_gt(), _gt(), SCHEMA) == 1.0


def test_returns_float_by_default():
    out = json_anls_score(_gt(), _gt(), SCHEMA)
    assert isinstance(out, float)


def test_return_key_scores_gives_breakdown():
    score, node = json_anls_score(_gt(), _gt(), SCHEMA, return_key_scores=True)
    assert score == 1.0
    assert isinstance(node, ScoreNode)
    assert "total" in node.children


def test_schema_none_defaults_to_string():
    # both sides identical strings -> perfect under default string comparator
    gt = {"a": "hello", "b": "world"}
    assert json_anls_score(gt, gt) == 1.0


def test_score_from_text_strips_code_fence():
    raw = '```json\n{"currency": "USD"}\n```'
    assert score_from_text({"currency": "USD"}, raw, {"currency": "categorical"}) == 1.0


def test_score_from_text_unparseable_is_zero():
    assert score_from_text(_gt(), "the model refused to answer", SCHEMA) == 0.0


def test_score_from_text_unparseable_breakdown():
    score, node = score_from_text(
        _gt(), "no json here", SCHEMA, return_key_scores=True
    )
    assert score == 0.0 and node.note == "unparseable prediction"
