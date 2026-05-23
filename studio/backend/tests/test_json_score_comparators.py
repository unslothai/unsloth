# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from eval.json_score.comparators import get_comparator


def test_money_exact_match():
    cmp = get_comparator("money")
    assert cmp(1234.56, 1234.56) == 1.0


def test_money_proportional_difference():
    cmp = get_comparator("money")
    # 1 - |100-90|/max(100,90) = 1 - 10/100 = 0.9
    assert abs(cmp(100, 90) - 0.9) < 1e-9


def test_money_both_zero_is_perfect():
    cmp = get_comparator("money")
    assert cmp(0, 0) == 1.0


def test_money_zero_vs_nonzero_is_zero():
    cmp = get_comparator("money")
    assert cmp(0, 5) == 0.0


def test_money_within_abs_tol():
    cmp = get_comparator("money", abs_tol=0.5)
    assert cmp(100.0, 100.4) == 1.0


def test_money_within_rel_tol():
    cmp = get_comparator("money", rel_tol=0.05)
    assert cmp(100.0, 96.0) == 1.0  # 4% diff <= 5%


def test_money_parses_currency_string():
    cmp = get_comparator("money")
    assert cmp("$1,234.56", 1234.56) == 1.0


def test_money_uncoercible_is_zero():
    cmp = get_comparator("money")
    assert cmp("abc", 10) == 0.0


def test_numeric_is_money_alias():
    assert get_comparator("numeric")(100, 90) == get_comparator("money")(100, 90)
