# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The linearity guard four tests hand their verdict to, checked on paths whose cost is known.

`assert_linear` decides `test_update_release_notes.py`, `test_tool_xml_strip.py`,
`test_rag_store.py` and `test_pr5624_regressions.py`. It is a timing guard, so both of its
failure modes are silent: too loose and a quadratic path ships, too tight and an unrelated
branch goes red for someone else's load. tests/_shared/growth.py's own docstring records the
second one happening three times, and it happened again on unslothai/unsloth#11152 at 7.2x
after the paired-median form was supposed to have ended it.

So the predicate gets rows of its own, the same way `test_cache_budget_discipline.py` and
`test_windows_amd_gpu_scan_fallback.py` test the readers their guards hand the verdict to.

Cost is DECLARED, not spent. Every row drives `growth` with a fake clock that advances by
exactly what the shape says, so a "quadratic" path here is quadratic to the last digit and
nothing depends on the scheduler. Sleeping for these shapes instead would have made a test
of a flakiness guard flaky in the same way the guard was, which is the failure being fixed
rather than a way to check it: a nominal 4 ms leg that the runner stretches to 7 ms moves
the ratio across the bar and the row means nothing.
"""

from __future__ import annotations

import pytest

from growth import assert_linear, growth  # tests/_shared, on sys.path via tests/conftest.py


class FakeClock:
    """A `perf_counter` that only moves when something says how much time it took.

    `growth` reads the clock either side of `run`, so a `run` that advances this by the
    cost of its input makes the measured elapsed exactly that cost.
    """

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
    """A `run` costing `_UNIT * len(text) ** exponent` seconds, plus a stall on chosen calls.

    `stalls` maps a call index (0-based, counting every leg in order) to extra seconds, which
    is how contention is stated exactly: the real thing lands inside one leg of one pair, and
    because the big leg runs `factor`x longer it is the one that usually catches it.
    """
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
    """#11152's failure, reproduced exactly and then not reproduced.

    Three pairs is few enough that two stalled big legs move the median over the bar. The
    stalls are placed on call indices 1 and 3, the big legs of the first two pairs, and are
    sized so the first median reads well above 6. A linear path must still pass, because
    the re-measurement is taken on a sample with no stall in it.
    """
    run, clock, calls = _shaped(1.0, stalls = {1: 40.0, 3: 40.0})
    assert assert_linear(run, _build, "stalled but linear", 2, clock = clock) == _build(8)
    # It really did take the retry: three pairs is six legs, so anything past six is the
    # second sample. A row that passed on the first reading would prove nothing about it.
    assert calls["n"] > 6, "the first sample did not trip the bar, so the retry was not exercised"


def test_the_retry_is_not_a_second_chance_for_a_quadratic_path():
    """The half that must NOT be weakened: re-measuring may only forgive noise, not a shape.

    Same stalls as the row above, on a genuinely quadratic path. Every pair of a quadratic
    path measures ~`factor ** 2`, so the larger second sample has to come back over the bar
    too. If it does not, the retry has turned the guard off.
    """
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
    """`assert_linear`'s 60s backstop reads this, and one stalled run must not trip it.

    Contention only adds, so the minimum big leg is the closest that size got to its own
    cost. A backstop reading the worst would fire on a runner that stalled once.
    """
    run, clock, _ = _shaped(1.0, stalls = {1: 50.0})
    _, big, _, _ = growth(run, _build, 2, repeats = 3, clock = clock)
    assert big == pytest.approx(_UNIT * 8), f"the stalled leg was the big time: {big}"


def test_a_path_slow_enough_to_trip_the_backstop_fails_on_the_backstop():
    """The other arm of the backstop: too slow to measure still has to fail, and say so."""
    run, clock, _ = _shaped(1.0, stalls = {1: 120.0})
    with pytest.raises(AssertionError, match = "path took"):
        assert_linear(run, _build, "glacial", 2, clock = clock)


def test_a_confirmation_that_would_outlast_the_job_is_not_started():
    """The retry must not cost more than the runner will wait, or the message is a timeout.

    Seven pairs of a big leg just under the 60s backstop is ~420s, and the four pytest
    invocations in .github/workflows/studio-backend-ci.yml pass `--timeout=330`: the worker
    is killed mid-confirmation and reports a bare timeout, so the shape that was measured
    never reaches anyone. Shaped here as a big leg of 40s with a ratio over the bar, which
    is the regression this describes and not a contended runner. The row pins BOTH halves:
    it fails as a linearity failure naming the first sample, and it does not pay for a
    second one -- three pairs is six legs, and nothing past six may run.
    """
    run, clock, calls = _shaped(1.0, stalls = {1: 40.0, 3: 40.0, 5: 40.0})
    with pytest.raises(AssertionError, match = "does not fit in"):
        assert_linear(run, _build, "slow and superlinear", 2, clock = clock)
    assert calls["n"] == 6, f"a confirmation it cannot afford was started anyway: {calls['n']}"


def test_a_confirmation_it_can_only_partly_afford_is_still_taken():
    """The other arm: short of seven pairs is a weaker vote, not a reason to skip the retry.

    Big legs of 30s put 90s of the 240s total into the first sample, so 150s is left and a
    30s pair affords four of them, not seven. The path is linear once the stalls stop, so it
    must still pass -- on four pairs, rather than on the ratio being forgiven. Without this
    the row above could be satisfied by refusing every confirmation that is not free.
    """
    run, clock, calls = _shaped(1.0, stalls = {1: 30.0, 3: 30.0, 5: 30.0})
    assert assert_linear(run, _build, "affordable in part", 2, clock = clock) == _build(8)
    # Six legs of the first sample, then four pairs rather than seven.
    assert calls["n"] == 6 + 8, f"the confirmation ran {(calls['n'] - 6) // 2} pairs"


def test_a_confirmation_sized_from_a_mixed_estimate_still_stops_at_the_budget():
    """Sizing the retry from the first sample is an estimate; the budget has to be measured.

    The estimate reads the BEST big leg and the MEDIAN ratio, and those come from different
    pairs. A first sample of 59s, 59s and 1s big legs against 1s small legs therefore offers
    a one-second pair cost alongside a ratio of 59, and authorises all seven pairs. If the
    confirmation's own legs then arrive at 40s, seven of them is about 290s, which on top of
    the first sample is past the `--timeout=330` those CI invocations pass.

    So the elapsed deadline inside the sample is what stops it, not the prediction made
    before it started: the loop gives up after two of the seven pairs it was authorised and
    the run fails with a sentence about an unfinished confirmation rather than with the
    runner's timeout.
    """
    stalls = {0: 0.998, 1: 58.992, 2: 0.998, 3: 58.992, 4: 0.998, 5: 0.992}
    for index in range(6, 6 + 2 * 7):
        stalls[index] = 0.998 if index % 2 == 0 else 39.992
    run, clock, calls = _shaped(1.0, stalls = stalls)
    with pytest.raises(AssertionError, match = "confirmation stopped after"):
        assert_linear(run, _build, "mixed estimate", 2, clock = clock)
    assert calls["n"] == 6 + 4, f"the confirmation ran on past its budget: {calls['n']}"


def test_the_budget_is_asked_before_a_pair_rather_than_after_it():
    """A bound tested once a pair is already home is not a bound on that pair.

    First-sample big legs of 13s, 20s and 20s against 1s small legs cost 56s, so 184s of the
    240s total is left, and at the sizing's estimated pair cost that authorises all seven.
    The confirmation's own legs then arrive at 59s, just inside the per-leg backstop.

    Asked AFTER each pair, the fourth one runs: 177s elapsed is under 184 at the moment the
    third finishes, so the sample reaches 236s and blows a bound it was never tested against
    while inside it. Reserving room for one more pair at the cost of the worst seen stops it
    at three, for 177s, and the whole test lands at 233s.
    """
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
    """A confirmation that stopped after one pair is not a reading, however low its ratio.

    The shape: the first sample is over the bar, then the retry's first pair has BOTH legs
    stalled -- the big one past the 60s backstop, which cuts the loop, and the small one
    enough to drag that pair's ratio under the tolerance. Reading the median of that one
    ratio, and taking the better of the two samples' big times, accepted a superlinear path
    on a sample that never finished. The stalls here are sized so the surviving ratio really
    is under 6, so the row fails for the abort and not for the ratio.
    """
    run, clock, _ = _shaped(2.0, stalls = {1: 40.0, 3: 40.0, 6: 30.0, 7: 70.0})
    with pytest.raises(AssertionError, match = "while re-measuring|confirmation stopped after"):
        assert_linear(run, _build, "aborted confirmation", 2, clock = clock)
