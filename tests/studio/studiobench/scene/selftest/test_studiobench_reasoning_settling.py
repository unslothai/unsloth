# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`reasoning_toggle` must settle on a quiet DOM and say plainly when it could not.

These drive the real `reasoning_toggle` with a stubbed page, so they fail on the tree before the
fix for the RIGHT reason -- a wrong value or a wrong message -- rather than by failing to import.

Two published wrong numbers came out of this one action:

  * `highlight_spans_while_open` read on the frame `data-state` flipped gave 74,917 on the
    measured-height arm and 44,075 on the grid-rows arm, an apparent 41% reduction. Settled, both
    arms read 74,250. Its null control was 0.0% and could not have caught it, because a null runs
    one bundle against itself and the skew cancels.
  * `open_ms` terminated on that same flip, and above the 100K rung it never terminated at all, so
    the metric silently became 100K-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tests.studio.studiobench.scene.actions import reasoning_toggle


@dataclass
class _Page:
    """A page that returns a canned reply and remembers how it was called.

    DELIBERATELY TOLERANT of both the old `timeoutMs` argument and the new
    `[timeoutMs, quietFrames]` pair. If the stub rejected the old shape, every test below would
    fail on the pre-fix tree for that reason alone, and none of them would demonstrate the
    substantive defect they exist to pin. The argument shape has its own test.
    """

    reply: dict
    last_arg: Any = None

    def evaluate(
        self,
        script: str,
        arg: Any = None,
    ) -> Any:
        self.last_arg = arg
        return self.reply


@dataclass
class _Ctx:
    page: _Page

    def log(self, *_a: Any, **_k: Any) -> None:
        return None


def _run(reply: dict):
    return reasoning_toggle(_Ctx(_Page(reply)))


def test_the_settle_is_told_how_quiet_is_quiet_enough():
    """A settle with no quiet-frame requirement is not a settle, it is a timeout."""
    page = _Page(_settled_reply())
    reasoning_toggle(_Ctx(page))
    assert isinstance(page.last_arg, list) and len(page.last_arg) == 2, (
        f"reasoning_toggle must pass [timeoutMs, quietFrames]; got {page.last_arg!r}. Terminating "
        f"on a state attribute alone is what made the census 41% wrong on one arm."
    )
    assert page.last_arg[1] >= 1


def _settled_reply(**over: Any) -> dict:
    base = {
        "ran": True,
        "panes": 10,
        "before": 0,
        "openCount": 10,
        "afterClose": 0,
        "spansOpen": 74250,
        "spansOpenReason": None,
        "openMs": 1777.8,
        "closeMs": 596.9,
        "openCensored": False,
        "closeCensored": False,
        "openCensoredReason": None,
        "closeCensoredReason": None,
        "openFrames": 42,
        "closeFrames": 18,
        "openStateReachedMs": 900.0,
        "quietFramesRequired": 4,
        "timeoutMs": 8000,
    }
    base.update(over)
    return base


def test_a_settled_toggle_reports_its_census_and_passes():
    res = _run(_settled_reply())
    assert res.ran and res.expect_ok
    assert res.reason is None
    assert res.expect["highlight_spans_while_open"] == 74250
    assert res.expect["settled"] is True
    assert res.timings == {"open_ms": 1777.8, "close_ms": 596.9}


def test_an_unsettled_census_is_withheld_rather_than_guessed():
    """Silence beats a confident wrong answer.

    This is the 44,075 case: the panes are open, the state attribute has flipped, and the spans
    are still arriving. The old code returned the half-mounted count as though it were the answer.
    """
    res = _run(
        _settled_reply(
            spansOpen = None,
            spansOpenReason = "the span census was still changing when the budget ran out",
            openMs = None,
            openCensored = True,
            openCensoredReason = "the span census was still changing when the budget ran out",
        )
    )
    assert res.expect["highlight_spans_while_open"] is None, (
        "a span count read from a DOM that was still mounting was reported as a census. That is "
        "the reading that produced an apparent 41% reduction between two trees that in fact "
        "mount the same document."
    )
    assert res.expect["settled"] is False
    assert res.expect["highlight_spans_while_open_reason"]


def test_a_censored_timing_is_absent_not_none():
    """A `None` in `timings` is dropped downstream and becomes an invisible missing cell."""
    res = _run(
        _settled_reply(openMs = None, openCensored = True, openCensoredReason = "never went quiet")
    )
    assert "open_ms" not in res.timings, (
        "a censored timing must not be carried as a key at all; downstream it is dropped for "
        "being non-numeric and the metric silently loses that cell"
    )
    assert res.timings["close_ms"] == 596.9
    assert res.expect["open_censored"] is True
    assert res.expect["open_censored_reason"] == "never went quiet"


def test_the_failure_reason_names_the_clause_that_actually_failed():
    """The exact message this replaces described a PASSING condition.

    Observed repeatedly at 500K and 1M:
        `ran EXPECT FAILED -- 16 of 16 panes opened and 0 were still open after collapsing`
    Every clause of that describes success. The real cause was always a censored `open_ms`.
    """
    res = _run(
        _settled_reply(
            panes = 16,
            openCount = 16,
            afterClose = 0,
            openMs = None,
            openCensored = True,
            openCensoredReason = "the open count reached 16 but the span census kept changing",
        )
    )
    assert res.expect_ok is False
    assert res.reason is not None
    assert "censored" in res.reason, (
        f"the reason must name the clause that failed; got {res.reason!r}, which is the old "
        f"message describing a passing condition"
    )
    assert "16 of 16 panes opened" not in res.reason


def test_a_genuine_pane_failure_still_names_the_panes():
    res = _run(_settled_reply(openCount = 9, panes = 10))
    assert res.expect_ok is False
    assert "9 of 10" in res.reason


def test_panes_left_open_after_collapse_are_reported():
    res = _run(_settled_reply(afterClose = 3))
    assert res.expect_ok is False
    assert "still open after collapsing" in res.reason


def test_the_ruler_resolution_is_reported_beside_the_timing():
    """`open_ms` is quantised to the paint interval, so the frame count has to travel with it."""
    res = _run(_settled_reply())
    assert res.expect["open_frames"] == 42
    assert res.expect["quiet_frames_required"] == 4
    assert res.expect["open_state_reached_ms"] == 900.0, (
        "how much of open_ms was spent AFTER the state attribute flipped is the whole difference "
        "between the two arms, so it has to be visible in the payload"
    )
