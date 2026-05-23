# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from eval.json_score import json_anls_score

anls_star = pytest.importorskip("anls_star")


# Only documents where global-leaf-equal aggregation and anls_star's scheme
# agree: identical docs, a flat single-level dict with one differing value, and
# reordered lists. Hallucinated keys and non-perfect nested objects are
# intentionally NOT cross-checked (this metric diverges there by design).
REFERENCE_CASES = [
    ({"a": "hello", "b": "world"}, {"a": "hello", "b": "world"}),       # 1.0
    ({"a": "hello", "b": "world"}, {"a": "hello", "b": "earth"}),       # flat, b differs
    (
        {"items": ["alpha", "beta", "gamma"]},
        {"items": ["gamma", "alpha", "beta"]},                          # reorder -> 1.0
    ),
    (
        {"v": {"name": "Acme", "id": "X1"}, "n": "Bob"},
        {"v": {"name": "Acme", "id": "X1"}, "n": "Bob"},                # nested, perfect
    ),
]


@pytest.mark.parametrize("gt, pred", REFERENCE_CASES)
def test_matches_anls_star_on_agreeing_documents(gt, pred):
    ours = json_anls_score(gt, pred)  # schema=None -> all string/ANLS
    theirs = anls_star.anls_score(gt, pred)
    assert abs(ours - theirs) < 1e-6, f"ours={ours} theirs={theirs}"


def test_end_to_end_invoice_breakdown():
    schema = {
        "total": {"type": "money"},
        "currency": "categorical",
        "issue_date": {"type": "date"},
        "vendor": {"name": "string", "vat_id": "categorical"},
        "line_items": [{"desc": "string", "qty": "numeric", "price": "money"}],
    }
    gt = {
        "total": 100,
        "currency": "USD",
        "issue_date": "2024-01-15",
        "vendor": {"name": "Acme Inc", "vat_id": "X1"},
        "line_items": [
            {"desc": "Apple", "qty": 1, "price": 40},
            {"desc": "Banana", "qty": 2, "price": 60},
        ],
    }
    pred = {
        "total": 90,                       # money: 1 - 10/100 = 0.9
        "currency": "EUR",                 # categorical wrong: 0.0
        "issue_date": "Jan 15, 2024",      # date correct: 1.0
        "vendor": {"name": "Acme Inc", "vat_id": "X1"},  # 1.0, 1.0 (2 leaves)
        "line_items": [
            {"desc": "Apple", "qty": 1, "price": 40},    # perfect
            {"desc": "Banana", "qty": 2, "price": 60},   # perfect
        ],
    }
    # Per-child (score, n_leaves) contributions to the top object:
    #   total       (0.9, 1) -> 0.9
    #   currency    (0.0, 1) -> 0.0
    #   issue_date  (1.0, 1) -> 1.0
    #   vendor      (1.0, 2) -> 2.0
    #   line_items  (1.0, 2) -> 2.0   (2 perfect slots)
    # sum = 5.9 over n_leaves = 7
    expected = 5.9 / 7
    score, node = json_anls_score(gt, pred, schema, return_key_scores=True)
    assert abs(score - expected) < 1e-9
    assert node.children["currency"].score == 0.0
    assert abs(node.children["total"].score - 0.9) < 1e-9
