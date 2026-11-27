# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import math
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from core.inference import tools


def _reference(text, token_budget):
    spent = 0.0
    index = 0
    length = len(text)
    while index < length:
        start = index
        if text[index] == "%" and tools._HEX_PAIR_RE.match(text, index + 1):
            spent += 3.0
            index += 3
        else:
            spent += 1.0 if ord(text[index]) > 127 else 0.25
            index += 1
        if spent > token_budget:
            return start
    return length


_ALPHABETS = ("abcXYZ 09.,", "%", "%2F", "%zz", "%a", "中é\U0001f600")


def _random_text(rng):
    pieces = []
    for _ in range(rng.randint(0, 10)):
        weights = (8, 1, 1, 1, 1, 1)
        alphabet = rng.choices(_ALPHABETS, weights = weights)[0]
        if alphabet in ("%", "%2F", "%zz", "%a"):
            pieces.append(alphabet)
        else:
            pieces.append("".join(rng.choice(alphabet) for _ in range(rng.choice((1, 4, 17, 200)))))
    return "".join(pieces)


def _budgets(rng, text):
    n = len(text)
    out = [0.0, 0.25, 0.5, 0.24, 0.26, -0.0, -0.25, -1.0, float("nan"), float("inf"), -float("inf")]
    out += [2.0**50, 2.0**50 - 1, 1e300]
    out += [n / 4, n / 4 - 0.25, n / 4 + 0.25, (n - 1) / 4, (n + 1) / 4]
    out += [rng.uniform(0, n / 4 + 2) for _ in range(6)]
    out += [rng.randint(0, n + 2) / 4 for _ in range(6)]
    return out


@pytest.mark.parametrize("seed", range(20))
def test_matches_reference_loop(seed):
    rng = random.Random(seed)
    for _ in range(100):
        text = _random_text(rng)
        for budget in _budgets(rng, text):
            assert tools._dense_prefix_chars(text, budget) == _reference(text, budget), (
                text,
                budget,
            )


def test_escape_on_and_past_the_cut():
    base = "a" * 40
    for position in range(0, 44):
        for escape in ("%41", "%4", "%", "%%41"):
            text = base[:position] + escape + base[position:]
            for quarter in range(0, 4 * len(text) + 8):
                budget = quarter / 4
                assert tools._dense_prefix_chars(text, budget) == _reference(text, budget)
                budget = math.nextafter(budget, 0.0) if quarter else budget
                assert tools._dense_prefix_chars(text, budget) == _reference(text, budget)


def test_large_text_small_budget():
    text = "x" * 2_000_000
    assert tools._dense_prefix_chars(text, 1.0) == 4
    assert tools._dense_prefix_chars(text, 1000.0) == 4000
    assert tools._dense_prefix_chars(text, 1e9) == len(text)
    escaped = text[:3999] + "%41" + text
    assert tools._dense_prefix_chars(escaped, 1000.0) == _reference(escaped, 1000.0) == 3999


def test_ascii_text_is_answered_without_a_per_character_scan(monkeypatch):
    # The loop calls ord() once per plain character; a module global shadows the builtin.
    calls = []

    def counting_ord(char):
        calls.append(char)
        return ord(char)

    monkeypatch.setattr(tools, "ord", counting_ord, raising = False)
    text = "a" * 10_000
    assert tools._dense_prefix_chars(text, 1_000.0) == 4_000
    assert tools._dense_prefix_chars(text, 1e9) == 10_000
    assert calls == []
    escaped = "a%41" + "a" * 100
    assert tools._dense_prefix_chars(escaped, 10.0) == _reference(escaped, 10.0)
    assert calls == []
    # Budgets outside the guarded range still take the loop, so the hook does count.
    assert tools._dense_prefix_chars(escaped, float("nan")) == _reference(escaped, float("nan"))
    assert calls


def test_other_numeric_budgets_match():
    for text in ("", "abc", "a" * 50, "ab%41cd" * 5, "x" * 10 + "中"):
        for budget in (
            Fraction(9, 4),
            Fraction(1, 3),
            Decimal("2.25"),
            Decimal("0.3"),
            True,
            False,
            3,
            0,
        ):
            assert tools._dense_prefix_chars(text, budget) == _reference(text, budget)


def test_str_subclass_keeps_the_scanner():
    class Text(str):
        def isascii(self):
            raise AssertionError("the scanner never calls isascii")

    assert tools._dense_prefix_chars(Text("abcdef"), 1.0) == _reference("abcdef", 1.0) == 4


_PIECES = (
    "a",
    "Z9 .,|\n\r",
    "%",
    "%4",
    "%41",
    "%fF",
    "%zz",
    "%%41",
    "%41%42%43",
    "中",
    "é",
    "é",
    "\U0001f600",
    "\ud800",
    "\udfff",
    "█▌━",
    "\x7f\x80",
)


def _mixed_text(rng, size):
    return "".join(rng.choice(_PIECES) * rng.choice((1, 1, 2, 7)) for _ in range(size))


@pytest.mark.parametrize("seed", range(12))
def test_charged_runs_match_reference_on_every_budget(seed):
    rng = random.Random(1000 + seed)
    for _ in range(40):
        text = _mixed_text(rng, rng.randint(0, 24))
        if rng.random() < 0.3:
            text += rng.choice(("%", "%4", "%41", "中"))
        for quarter in range(0, 4 * len(text) + 13):
            budget = quarter / 4
            assert tools._dense_prefix_chars(text, budget) == _reference(text, budget), (
                text,
                budget,
            )
            below = math.nextafter(budget, -1.0)
            assert tools._dense_prefix_chars(text, below) == _reference(text, below), (text, below)
            assert tools._dense_prefix_chars(text, quarter) == _reference(text, quarter), (
                text,
                quarter,
            )


def test_long_dense_text_matches_reference():
    rng = random.Random(7)
    text = _mixed_text(rng, 4000)
    for budget in [0.0, 1.0, 2.75, 3.0] + [rng.uniform(0, len(text)) for _ in range(30)]:
        assert tools._dense_prefix_chars(text, budget) == _reference(text, budget), budget


def test_progress_output_is_answered_without_a_per_character_scan(monkeypatch):
    calls = []

    def counting_ord(char):
        calls.append(char)
        return ord(char)

    monkeypatch.setattr(tools, "ord", counting_ord, raising = False)
    bars = "".join(
        f"{p:3d}%|{'█' * (p // 10)}{' ' * (10 - p // 10)}| {p}/100 [00:01<00:00]\r"
        for p in range(101)
    )
    escaped = "q=%E4%B8%AD%20x " * 50
    expected = {
        (text, budget): _reference(text, budget)
        for text in (bars, escaped, bars + escaped)
        for budget in (10.0, 250.5, 1e6)
    }
    calls.clear()
    for (text, budget), want in expected.items():
        assert tools._dense_prefix_chars(text, budget) == want
    assert calls == []


def test_escape_runs_match_reference_on_every_budget():
    for text in (
        "%E4%B8%AD" * 40,
        "/wiki/%E4%B8%AD%E6%96%87%zz%%41%4" + "%41" * 30 + "%",
        "%41中%42é%%43" * 9,
    ):
        for quarter in range(0, 4 * len(text) + 13):
            for budget in (quarter / 4, math.nextafter(quarter / 4, -1.0), quarter):
                assert tools._dense_prefix_chars(text, budget) == _reference(text, budget), (
                    text,
                    budget,
                )


def test_long_percent_runs_match_reference():
    for text in ("%" * 5000 + "a", "%" * 3000 + "%41" * 500 + "中" * 50, ("%%4" + "%41" * 7) * 300):
        for budget in (0.0, 0.25, 10.0, 99.75, 750.5, 2000.0, 1e9):
            assert tools._dense_prefix_chars(text, budget) == _reference(text, budget), budget
