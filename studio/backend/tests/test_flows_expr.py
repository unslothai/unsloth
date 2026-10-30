# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

from core.flows.expr import ExprError, Prng, apply_sets, eval_expr, roll_dice, seed_from

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "flows_expr_cases.json").read_text(encoding = "utf-8")
)


@pytest.mark.parametrize("case", FIXTURES["cases"], ids = [c["name"] for c in FIXTURES["cases"]])
def test_shared_cases(case):
    rng = Prng(case["seed"]) if "seed" in case else None
    if case.get("error"):
        with pytest.raises(ExprError):
            eval_expr(case["expr"], case["vars"], rng)
    else:
        assert eval_expr(case["expr"], case["vars"], rng) == case.get("expected")


def test_prng_and_seed_fixtures():
    for p in FIXTURES["prng"]:
        rng = Prng(p["seed"])
        assert [rng.random() for _ in p["values"]] == p["values"]
        assert rng.state == p["state_after"]
    for s in FIXTURES["seed_from"]:
        assert seed_from(*s["parts"]) == s["expected"]


def test_budget_and_depth():
    deep = 1
    for _ in range(80):
        deep = {"op": "+", "args": [deep, 1]}
    with pytest.raises(ExprError):
        eval_expr(deep)
    wide = {"op": "+", "args": [1] * 20_000}
    with pytest.raises(ExprError):
        eval_expr(wide)


def test_never_evaluates_source():
    assert eval_expr("__import__('os').system('true')") == "__import__('os').system('true')"
    with pytest.raises(ExprError):
        eval_expr({"op": "__class__", "args": []})


def test_apply_sets_reads_old_values():
    out = apply_sets({"a": {"var": "b"}, "b": {"var": "a"}}, {"a": 1, "b": 2})
    assert out == {"a": 2, "b": 1}


def test_dice_range():
    rng = Prng(5)
    for _ in range(200):
        total, rolls = roll_dice("2d6+1", rng)
        assert 3 <= total <= 13 and len(rolls) == 2
    with pytest.raises(ExprError):
        roll_dice("1000d6", rng)


def test_round_and_multiply_stay_bounded():
    import time

    start = time.monotonic()
    for expr in (
        {"op": "round", "args": [1, 1_000_000]},
        {"op": "round", "args": [1, -10_000_000]},
        {"op": "round", "args": [1e300, 100]},
        {"op": "*", "args": [{"var": "x"}] * 5000},
    ):
        with pytest.raises(ExprError):
            eval_expr(expr, {"x": 10**300})
    assert time.monotonic() - start < 1.0
    assert eval_expr({"op": "round", "args": [1.23456, 2]}) == 1.23
    assert eval_expr({"op": "*", "args": [2, 3, 4]}) == 24
