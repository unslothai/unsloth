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
    cmp = get_comparator("money", abs_tol = 0.5)
    assert cmp(100.0, 100.4) == 1.0


def test_money_within_rel_tol():
    cmp = get_comparator("money", rel_tol = 0.05)
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
    with pytest.raises(ValueError, match = "Unknown comparator"):
        get_comparator("nope")


def test_get_comparator_bad_param_raises():
    with pytest.raises(ValueError, match = "Invalid params"):
        get_comparator("money", bogus = 1)


def test_categorical_exact():
    cmp = get_comparator("categorical")
    assert cmp("USD", "USD") == 1.0
    assert cmp("USD", "EUR") == 0.0


def test_categorical_case_insensitive_and_strip():
    cmp = get_comparator("categorical", case_insensitive = True, strip = True)
    assert cmp("USD", " usd ") == 1.0


def test_categorical_non_string_equality():
    cmp = get_comparator("categorical")
    assert cmp(True, True) == 1.0
    assert cmp(5, 5) == 1.0


def test_string_identical():
    cmp = get_comparator("string")
    assert cmp("Acme Inc", "Acme Inc") == 1.0


def test_string_below_threshold_is_zero():
    cmp = get_comparator("string", threshold = 0.5)
    assert cmp("abcdefgh", "zzzzzzzz") == 0.0


def test_string_high_similarity_kept():
    cmp = get_comparator("string", threshold = 0.5)
    assert cmp("Acme Incorporated", "Acme Incorporatd") > 0.9


def test_string_handles_none_as_empty():
    cmp = get_comparator("string")
    assert cmp(None, None) == 1.0


def test_date_format_agnostic_equal():
    cmp = get_comparator("date")
    assert cmp("2024-01-15", "Jan 15, 2024") == 1.0
    assert cmp("2024-01-15", "15 January 2024") == 1.0


def test_date_different_days():
    cmp = get_comparator("date")
    assert cmp("2024-01-15", "2024-01-16") == 0.0


def test_date_day_tolerance():
    cmp = get_comparator("date", day_tol = 1)
    assert cmp("2024-01-15", "2024-01-16") == 1.0


def test_date_granularity_month():
    cmp = get_comparator("date", granularity = "month")
    assert cmp("2024-01-15", "2024-01-28") == 1.0
    assert cmp("2024-01-15", "2024-02-15") == 0.0


def test_date_unparseable_is_zero():
    cmp = get_comparator("date")
    assert cmp("not a date", "2024-01-15") == 0.0


def test_date_unknown_granularity_raises():
    with pytest.raises(ValueError, match = "Unknown granularity"):
        get_comparator("date", granularity = "week")


def test_date_accepts_date_objects():
    from datetime import date, datetime

    cmp = get_comparator("date")
    assert cmp(date(2024, 1, 15), "2024-01-15") == 1.0
    assert cmp(datetime(2024, 1, 15, 9, 30), date(2024, 1, 15)) == 1.0


def test_date_year_granularity_ignores_day_tol():
    cmp = get_comparator("date", granularity = "year", day_tol = 999)
    assert cmp("2024-06-01", "2024-12-31") == 1.0
    assert cmp("2024-06-01", "2025-06-01") == 0.0


def test_string_none_and_empty_mix():
    cmp = get_comparator("string")
    assert cmp(None, "") == 1.0
    assert cmp(None, "x") == 0.0
