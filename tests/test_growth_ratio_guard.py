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

Cost is SIMULATED rather than spent: `run` sleeps for a duration computed from the input
size, so a "quadratic" path here is quadratic exactly and a whole file of these runs in a
couple of seconds. That is the point -- a test of a timing guard that had to really be slow
could only be run at sizes too small to separate the shapes.
"""

from __future__ import annotations

import time

import pytest

from growth import assert_linear, growth  # tests/_shared, on sys.path via tests/conftest.py


#: Small enough to keep the file fast, large enough to sit well above the timer's resolution
#: and above the 1e-4 floor `growth` clamps the small leg to.
_UNIT_SECONDS = 0.002


def _shaped(exponent: float, *, spikes: list[int] = None):
    """A `run` whose cost is `len(text) ** exponent`, optionally stalling on chosen calls.

    `spikes` names call indices (0-based, counting every leg) that additionally sleep long
    enough to look like a scheduler stall. That is how contention is reproduced deterministically:
    the real thing lands inside one leg of one pair, and because the big leg runs `factor`x
    longer it is the one that usually catches it.
    """
    spikes = set(spikes or [])
    calls = {"n": 0}

    def run(text: str) -> str:
        index = calls["n"]
        calls["n"] += 1
        time.sleep(_UNIT_SECONDS * (len(text) ** exponent))
        if index in spikes:
            time.sleep(_UNIT_SECONDS * 12)
        return text

    return run, calls


def _build(n: int) -> str:
    return "x" * n


def test_a_linear_path_passes():
    """The negative control. Without it every row below could pass by the guard never accepting."""
    run, _ = _shaped(1.0)
    assert assert_linear(run, _build, "linear", 2) == _build(8)


def test_a_quadratic_path_fails():
    """The positive control: the shape these guards exist to catch."""
    run, _ = _shaped(2.0)
    with pytest.raises(AssertionError, match = "is not linear"):
        assert_linear(run, _build, "quadratic", 2)


def test_a_stall_in_the_first_sample_does_not_fail_a_linear_path():
    """#11152's failure, reproduced and then not reproduced.

    Three pairs is few enough that one stalled big leg moves the median over the bar. The
    stall is placed on call index 1, which is the big leg of the first pair, and made large
    enough that the first median alone would read as superlinear. A linear path must still
    pass, because the re-measurement is taken on a quiet sample.
    """
    run, calls = _shaped(1.0, spikes = [1, 3])
    assert assert_linear(run, _build, "stalled but linear", 2) == _build(8)
    # It really did take the retry: three pairs is six legs, so anything past six is the
    # second sample. A row that passed on the first reading would prove nothing about it.
    assert calls["n"] > 6, "the first sample did not trip the bar, so the retry was not exercised"


def test_the_retry_is_not_a_second_chance_for_a_quadratic_path():
    """The half that must NOT be weakened: re-measuring may only forgive noise, not a shape.

    Same stalls as the row above, on a genuinely quadratic path. Every pair of a quadratic
    path measures ~`factor ** 2`, so the larger second sample has to come back over the bar
    too -- if it does not, the retry has turned the guard off.
    """
    run, _ = _shaped(2.0, spikes = [1, 3])
    with pytest.raises(AssertionError, match = "is not linear"):
        assert_linear(run, _build, "stalled and quadratic", 2)


def test_the_failure_message_names_both_samples():
    """A red run has to say it was measured twice, or the next person re-litigates the retry."""
    run, _ = _shaped(2.0)
    with pytest.raises(AssertionError) as excinfo:
        assert_linear(run, _build, "quadratic", 2)
    message = str(excinfo.value)
    assert "pairs, after" in message, message
    assert "quadratic is ~16" in message, message


def test_growth_reports_the_best_big_time_not_the_worst():
    """`assert_linear`'s 60s backstop reads this, and a stalled run must not trip it.

    Contention only adds, so the minimum big leg is the closest that size got to its own
    cost. A backstop reading the worst would fire on a runner that stalled once.
    """
    run, _ = _shaped(1.0, spikes = [1])
    _, big, _ = growth(run, _build, 2, repeats = 3)
    quiet = _UNIT_SECONDS * 8
    assert big < quiet + _UNIT_SECONDS * 6, f"the stalled leg was reported as the big time: {big}"
