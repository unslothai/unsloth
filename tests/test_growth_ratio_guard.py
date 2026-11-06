# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checks assert_linear on fake-clock paths of declared cost, since real sleeps would be flaky."""

from __future__ import annotations

import pytest

from growth import assert_linear, growth  # tests/_shared, on sys.path via tests/conftest.py


class FakeClock:
    """perf_counter stand-in that moves only when run reports its cost, so elapsed time is exact."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


#: Seconds per unit of cost. Small enough that a quadratic row stays well under the 60s
#: backstop `assert_linear` applies, so the quadratic rows fail on the RATIO, which is what
#: they are about, rather than on the backstop, which has its own row.
_UNIT = 0.001


def _shaped(exponent: float, *, stalls: dict[int, float] = None):
    """Fake run costing _UNIT * len(text) ** exponent seconds, with stalls added on chosen calls."""
    stalls = stalls or {}
    clock = FakeClock()
    calls = {"n": 0}

    def run(text: str) -> str:
        index = calls["n"]
        calls["n"] += 1
        clock.advance(_UNIT * len(text) ** exponent + stalls.get(index, 0.0))
        return text

    return run, clock, calls


def _build(n: int) -> str:
    return "x" * n


def test_a_linear_path_passes():
    """The negative control. Without it every row below could pass by the guard never accepting."""
    run, clock, _ = _shaped(1.0)
    assert assert_linear(run, _build, "linear", 2, clock = clock) == _build(8)


def test_a_quadratic_path_fails():
    """The positive control: the shape these guards exist to catch."""
    run, clock, _ = _shaped(2.0)
    with pytest.raises(AssertionError, match = "is not linear"):
        assert_linear(run, _build, "quadratic", 2, clock = clock)


def test_a_stall_in_the_first_sample_does_not_fail_a_linear_path():
    """Two stalled big legs push a three-pair median over the bar; a linear path must pass on retry."""
    run, clock, calls = _shaped(1.0, stalls = {1: 40.0, 3: 40.0})
    assert assert_linear(run, _build, "stalled but linear", 2, clock = clock) == _build(8)
    # Three pairs is six legs, so anything past six proves the retry ran.
    assert calls["n"] > 6, "the first sample did not trip the bar, so the retry was not exercised"


def test_the_retry_is_not_a_second_chance_for_a_quadratic_path():
    """A quadratic path must still fail on the retry, or the re-measurement has turned the guard off."""
    run, clock, _ = _shaped(2.0, stalls = {1: 40.0, 3: 40.0})
    with pytest.raises(AssertionError, match = "is not linear"):
        assert_linear(run, _build, "stalled and quadratic", 2, clock = clock)


def test_the_failure_message_names_both_samples():
    """A red run has to say it was measured twice, or the next person re-litigates the retry."""
    run, clock, _ = _shaped(2.0)
    with pytest.raises(AssertionError) as excinfo:
        assert_linear(run, _build, "quadratic", 2, clock = clock)
    message = str(excinfo.value)
    assert "pairs, after" in message, message
    assert "quadratic is ~16" in message, message


def test_growth_reports_the_best_big_time_not_the_worst():
    """growth reports the minimum big time: contention only adds, so one stall cannot trip the backstop."""
    run, clock, _ = _shaped(1.0, stalls = {1: 50.0})
    _, big, _, _ = growth(run, _build, 2, repeats = 3, clock = clock)
    assert big == pytest.approx(_UNIT * 8), f"the stalled leg was the big time: {big}"


def test_a_path_slow_enough_to_trip_the_backstop_fails_on_the_backstop():
    """The other arm of the backstop: too slow to measure still has to fail, and say so."""
    run, clock, _ = _shaped(1.0, stalls = {1: 120.0})
    with pytest.raises(AssertionError, match = "path took"):
        assert_linear(run, _build, "glacial", 2, clock = clock)


def test_the_first_sample_is_inside_the_budget_too():
    """The budget counts the first sample too, or three slow pairs run past --timeout=330."""
    stalls = {}
    for index in range(6):
        stalls[index] = 49.998 if index % 2 == 0 else 54.992
    run, clock, calls = _shaped(1.0, stalls = stalls)
    with pytest.raises(AssertionError, match = "first sample stopped after 2 of 3"):
        assert_linear(run, _build, "slow on both legs", 2, clock = clock)
    assert calls["n"] == 4, f"a pair that could not fit was started anyway: {calls['n']}"
    assert clock.now - 1000.0 < 330.0, f"the call outlasted the runner: {clock.now - 1000.0}"


def test_a_confirmation_that_would_outlast_the_job_is_not_started():
    """A confirmation that cannot fit the runner's timeout is not started, so no second sample runs."""
    run, clock, calls = _shaped(1.0, stalls = {1: 40.0, 3: 40.0, 5: 40.0})
    with pytest.raises(AssertionError, match = "does not fit in"):
        assert_linear(run, _build, "slow and superlinear", 2, clock = clock)
    assert calls["n"] == 6, f"a confirmation it cannot afford was started anyway: {calls['n']}"


def test_a_confirmation_it_can_only_partly_afford_is_still_taken():
    """A confirmation that can only afford four pairs must still run and pass, not be skipped."""
    run, clock, calls = _shaped(1.0, stalls = {1: 30.0, 3: 30.0, 5: 30.0})
    assert assert_linear(run, _build, "affordable in part", 2, clock = clock) == _build(8)
    assert calls["n"] == 6 + 8, f"the confirmation ran {(calls['n'] - 6) // 2} pairs"


def test_a_confirmation_sized_from_a_mixed_estimate_still_stops_at_the_budget():
    """The live elapsed clock stops a confirmation: the size estimate mixes legs from different pairs."""
    stalls = {0: 0.998, 1: 58.992, 2: 0.998, 3: 58.992, 4: 0.998, 5: 0.992}
    for index in range(6, 6 + 2 * 7):
        stalls[index] = 0.998 if index % 2 == 0 else 39.992
    run, clock, calls = _shaped(1.0, stalls = stalls)
    with pytest.raises(AssertionError, match = "confirmation stopped after"):
        assert_linear(run, _build, "mixed estimate", 2, clock = clock)
    assert calls["n"] == 6 + 4, f"the confirmation ran on past its budget: {calls['n']}"


def test_the_budget_is_asked_before_a_pair_rather_than_after_it():
    """The budget is checked before each pair, with room reserved for the worst pair seen so far."""
    stalls = {0: 0.998, 1: 12.992, 2: 0.998, 3: 19.992, 4: 0.998, 5: 19.992}
    for index in range(6, 6 + 2 * 7):
        stalls[index] = 0.0 if index % 2 == 0 else 58.992
    run, clock, calls = _shaped(1.0, stalls = stalls)
    with pytest.raises(AssertionError, match = "confirmation stopped after 3 of 7"):
        assert_linear(run, _build, "pair that would overrun", 2, clock = clock)
    assert calls["n"] == 6 + 6, f"a pair that could not fit was started anyway: {calls['n']}"
    assert (
        clock.now - 1000.0 < 330.0
    ), f"the whole test would outlast the runner: {clock.now - 1000.0}"


def test_an_aborted_confirmation_is_not_accepted_as_one():
    """A confirmation aborted after one pair is not a reading, even when its ratio looks linear."""
    run, clock, _ = _shaped(2.0, stalls = {1: 40.0, 3: 40.0, 6: 30.0, 7: 70.0})
    with pytest.raises(AssertionError, match = "while re-measuring|confirmation stopped after"):
        assert_linear(run, _build, "aborted confirmation", 2, clock = clock)
