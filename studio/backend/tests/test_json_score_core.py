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


from eval.json_score.core import _mismatch, _score_object


def test_object_all_correct():
    schema = normalize_schema({"total": "money", "currency": "categorical"})
    gt = {"total": 100, "currency": "USD"}
    pred = {"total": 100, "currency": "USD"}
    node = _score_object(gt, pred, schema, "string")
    assert node.score == 1.0 and node.n_leaves == 2


def test_object_one_field_wrong():
    schema = normalize_schema({"total": "money", "currency": "categorical"})
    gt = {"total": 100, "currency": "USD"}
    pred = {"total": 100, "currency": "EUR"}
    node = _score_object(gt, pred, schema, "string")
    assert node.score == 0.5  # one of two leaves correct


def test_object_missing_key_penalized_proportionally():
    schema = normalize_schema(
        {"total": "money", "vendor": {"name": "string", "vat": "categorical"}}
    )
    gt = {"total": 100, "vendor": {"name": "Acme", "vat": "X1"}}
    pred = {"total": 100}  # whole vendor subtree (2 leaves) missing
    node = _score_object(gt, pred, schema, "string")
    # 3 leaves total, 1 correct -> 1/3
    assert abs(node.score - (1 / 3)) < 1e-9 and node.n_leaves == 3


def test_object_hallucinated_key_costs_one_leaf():
    schema = normalize_schema({"total": "money"})
    gt = {"total": 100}
    pred = {"total": 100, "extra": "junk"}
    node = _score_object(gt, pred, schema, "string")
    # 1 correct leaf + 1 hallucinated zero-leaf -> 1/2
    assert node.score == 0.5 and node.n_leaves == 2


def test_object_empty_both_is_perfect():
    node = _score_object({}, {}, normalize_schema({}), "string")
    assert node.score == 1.0 and node.n_leaves == 0


def test_dispatch_infers_object_when_no_schema():
    node = _score({"a": "x"}, {"a": "x"}, None, "string")
    assert node.score == 1.0 and node.n_leaves == 1


def test_options_at_object_level():
    schema = normalize_schema({"name": "string"})
    gt = ({"name": "A"}, {"name": "B"})
    node = _score(gt, {"name": "B"}, schema, "string")
    assert node.score == 1.0 and node.matched_option == 1


def test_mismatch_schema_object_data_scalar():
    schema = normalize_schema({"a": "string", "b": "string"})
    node = _mismatch(7, "x", schema)
    assert node.score == 0.0 and node.note == "type mismatch"


from eval.json_score.core import _score_array


def _items_schema():
    return normalize_schema([{"desc": "string", "price": "money"}])


def test_array_order_invariant():
    schema = _items_schema()
    gt = [{"desc": "Apple", "price": 10}, {"desc": "Banana", "price": 20}]
    pred = [{"desc": "Banana", "price": 20}, {"desc": "Apple", "price": 10}]
    node = _score_array(gt, pred, schema, "string")
    assert node.score == 1.0 and node.n_leaves == 2


def test_array_extra_predicted_item_penalized():
    schema = _items_schema()
    gt = [{"desc": "Apple", "price": 10}]
    pred = [{"desc": "Apple", "price": 10}, {"desc": "Ghost", "price": 99}]
    node = _score_array(gt, pred, schema, "string")
    # 1 matched slot scores 1.0, divided by max(1, 2) = 2 -> 0.5
    assert node.score == 0.5 and node.n_leaves == 2


def test_array_missing_item_penalized():
    schema = _items_schema()
    gt = [{"desc": "Apple", "price": 10}, {"desc": "Banana", "price": 20}]
    pred = [{"desc": "Apple", "price": 10}]
    node = _score_array(gt, pred, schema, "string")
    assert node.score == 0.5 and node.n_leaves == 2


def test_array_empty_both_is_perfect():
    node = _score_array([], [], _items_schema(), "string")
    assert node.score == 1.0 and node.n_leaves == 0


def test_dispatch_infers_array_when_no_schema():
    node = _score([1, 2], [2, 1], None, "string")
    assert node.score == 1.0 and node.n_leaves == 2


def test_leaf_schema_on_container_is_mismatch():
    # a leaf comparator must not stringify a dict/list and Levenshtein its repr
    node = _score({"a": 1, "b": 2}, {"a": 1}, normalize_schema("string"), "string")
    assert node.score == 0.0 and node.note == "type mismatch"
    node = _score("scalar", [1, 2, 3], normalize_schema("string"), "string")
    assert node.score == 0.0 and node.note == "type mismatch"


def test_array_breakdown_includes_unmatched_items():
    schema = _items_schema()
    gt = [{"desc": "Apple", "price": 10}]
    pred = [{"desc": "Apple", "price": 10}, {"desc": "Ghost", "price": 99}]
    node = _score_array(gt, pred, schema, "string")
    notes = [c.note for c in node.children]
    assert "hallucinated" in notes  # the extra "Ghost" row is surfaced

    gt2 = [{"desc": "Apple", "price": 10}, {"desc": "Banana", "price": 20}]
    pred2 = [{"desc": "Apple", "price": 10}]
    node2 = _score_array(gt2, pred2, schema, "string")
    notes2 = [c.note for c in node2.children]
    assert "missing in prediction" in notes2  # the dropped "Banana" row is surfaced
