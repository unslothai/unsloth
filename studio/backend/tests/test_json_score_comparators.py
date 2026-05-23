# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from eval.json_score.comparators import _REGISTRY, get_comparator


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
    # alias must point at the same factory, not merely agree on one input
    assert _REGISTRY["numeric"] is _REGISTRY["money"]


def test_money_negative_values_symmetric():
    cmp = get_comparator("money")
    # same magnitudes/ratio as cmp(100, 90) -> 0.9
    assert abs(cmp(-100, -90) - 0.9) < 1e-9


def test_money_opposite_signs():
    cmp = get_comparator("money")
    # 1 - |(-50)-50| / max(50, 50) = 1 - 100/50 -> clamped to 0.0
    assert cmp(-50, 50) == 0.0


def test_money_negative_currency_string_keeps_sign():
    cmp = get_comparator("money")
    # regression: "-$1,234.56" must parse as negative, not positive
    assert cmp("-$1,234.56", -1234.56) == 1.0
    assert cmp("-$1,234.56", 1234.56) == 0.0


def test_money_scientific_notation_string():
    cmp = get_comparator("money")
    assert cmp("1.5e3", 1500) == 1.0


def test_money_non_finite_is_zero():
    cmp = get_comparator("money")
    assert cmp(float("nan"), 10) == 0.0
    assert cmp(float("inf"), 10) == 0.0


def test_money_bool_is_uncoercible():
    cmp = get_comparator("money")
    assert cmp(True, 1) == 0.0


def test_money_none_is_zero():
    cmp = get_comparator("money")
    assert cmp(None, 5) == 0.0


def test_get_comparator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown comparator"):
        get_comparator("nope")


def test_get_comparator_bad_param_raises():
    with pytest.raises(ValueError, match="Invalid params"):
        get_comparator("money", bogus=1)
