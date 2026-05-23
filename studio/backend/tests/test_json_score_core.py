# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from eval.json_score.core import (
    ScoreNode,
    _leaf_count,
    _score,
    _score_leaf,
)
from eval.json_score.schema import normalize_schema


def test_leaf_both_none_is_perfect():
    node = _score_leaf(None, None, "string", {})
    assert node.score == 1.0 and node.n_leaves == 1


def test_leaf_one_none_is_zero():
    node = _score_leaf("x", None, "string", {})
    assert node.score == 0.0 and node.note == "missing or hallucinated"


def test_leaf_uses_comparator():
    node = _score_leaf(100, 90, "money", {})
    assert abs(node.score - 0.9) < 1e-9 and node.n_leaves == 1


def test_leaf_count_scalar():
    assert _leaf_count(5, None) == 1


def test_leaf_count_object():
    assert _leaf_count({"a": 1, "b": {"c": 2, "d": 3}}, None) == 3


def test_leaf_count_array():
    # each array item counts as one slot (matches _score_array), not inner leaves
    assert _leaf_count([{"a": 1}, {"a": 2, "b": 3}], None) == 2


def test_leaf_count_tuple_uses_first_option():
    assert _leaf_count(("x", "y", "z"), None) == 1


def test_dispatch_leaf():
    node = _score(100, 90, normalize_schema("money"), "string")
    assert abs(node.score - 0.9) < 1e-9


def test_options_tuple_takes_best_alternative():
    schema = normalize_schema("string")
    node = _score(
        ("Acme Inc", "Acme Incorporated"), "Acme Incorporated", schema, "string"
    )
    assert node.score == 1.0 and node.matched_option == 1


def test_options_empty_tuple_is_zero():
    node = _score((), "anything", normalize_schema("string"), "string")
    assert node.score == 0.0 and node.note == "empty alternatives tuple"
