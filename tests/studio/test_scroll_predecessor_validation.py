# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A predecessor comparison must not be published on a scroll that did not happen.

`scroll_predecessor_probe.py` measures one gesture -- the harness's own SCROLL_JS -- under a set
of different predecessors, and reports the difference as the predecessor's cost. `checked()`
already proves the PREDECESSOR occurred. This file pins the other half: that the measured scroll,
the one thing every arm has in common and the only thing the comparison is actually made from,
occurred too.

The defect this file exists to keep out, previously live: `run_action`'s row was appended with no
validation whatsoever. SCROLL_JS returns null when the viewport is not in the DOM, returns
`settleMs: None` when the page never goes quiet inside the settle timeout, and returns whatever
distance it managed if the viewport refused to move. The probe's `med()` helper skips fields that
are absent or non-numeric, and only a RAISED exception marks an arm failed, so every one of those
produced a plausible-looking median, a full results table and exit code 0.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_probe():
    """Import the probe without a browser, the same way the sibling harness tests do.

    Deliberately NOT `pytest.importorskip`. This file has to run in the job that runs it, and a
    guard that skips is a guard that proves nothing; if the stubbing below ever stops working the
    correct outcome is a collection error someone has to look at, not a silent skip.
    """
    os.environ.setdefault("PW_ART_DIR", str(WORKDIR / "logs" / "heavy-thread-artifacts"))
    if "playwright.sync_api" not in sys.modules:
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError:
            package = types.ModuleType("playwright")
            module = types.ModuleType("playwright.sync_api")
            module.sync_playwright = None
            package.sync_api = module
            sys.modules["playwright"] = package
            sys.modules["playwright.sync_api"] = module
    import scroll_predecessor_probe

    return scroll_predecessor_probe


PROBE = _load_probe()


def good_row(**overrides) -> dict:
    """A scroll that completed: 20 steps of 400px, timed, and quiet afterwards."""
    row = {
        "name": "scroll",
        "ran": True,
        "gestureMs": 412.5,
        "settleMs": 88.0,
        "scrolledPx": 8000,
    }
    row.update(overrides)
    return row


# ── the row-level validation ──────────────────────────────────────────


def test_a_completed_scroll_is_accepted():
    """The control case. Without it every assertion below could be met by rejecting everything."""
    assert PROBE.scroll_row_problems(good_row()) == []
    assert PROBE.checked_scroll("nothing", good_row())["scrolledPx"] == 8000


def test_a_null_scroll_is_rejected():
    """SCROLL_JS returns null when the viewport is not in the DOM; `run_action` turns that into
    `ran: False` and a row of Nones, which `med()` silently drops."""
    problems = PROBE.scroll_row_problems(good_row(ran=False))
    assert problems, "a scroll that never ran was accepted"
    assert "null" in problems[0]


def test_a_scroll_that_never_went_quiet_is_rejected():
    """`__hv.quiet()` returns null when the settle timeout expires under continuous activity, so
    settle_ms would be a median of the repetitions that happened to settle."""
    problems = PROBE.scroll_row_problems(good_row(settleMs=None))
    assert any("never went quiet" in p for p in problems), problems


def test_a_scroll_with_no_timing_is_rejected():
    problems = PROBE.scroll_row_problems(good_row(gestureMs=None))
    assert any("no gesture duration" in p for p in problems), problems


def test_a_viewport_that_did_not_move_is_rejected():
    """Zero travel is the failure that looks most like a result: every timing field is present and
    numeric, and the row reads as a very fast scroll."""
    problems = PROBE.scroll_row_problems(good_row(scrolledPx=0))
    assert any("did not move" in p for p in problems), problems


def test_a_missing_travel_field_is_rejected():
    problems = PROBE.scroll_row_problems(good_row(scrolledPx=None))
    assert any("no travel" in p for p in problems), problems


def test_checked_scroll_raises_so_main_marks_the_arm_failed():
    """`main()` records a raise as a failed arm, keeps it out of the table and exits 1. Returning
    a problem list instead would leave the arm in the comparison."""
    with pytest.raises(RuntimeError) as exc:
        PROBE.checked_scroll("delete", good_row(ran=False))
    assert "delete" in str(exc.value)


# ── the cross-arm comparison ──────────────────────────────────────────


def arms(**travel) -> dict:
    return {n: {"scrolled_px": px} for n, px in travel.items()}


def test_arms_that_scrolled_the_same_distance_are_accepted():
    assert PROBE.skewed_arms(arms(nothing=8000, delete=8000, jump=7900)) == []


def test_an_arm_that_scrolled_a_different_distance_is_rejected():
    """A shorter thread under one predecessor means a shorter gesture, and the difference is then
    reported as that predecessor's cost."""
    skewed = PROBE.skewed_arms(arms(nothing=8000, delete=5200))
    assert len(skewed) == 1 and "delete" in skewed[0], skewed


def test_the_tolerance_is_slack_not_a_licence():
    """Pinned so widening it later is a visible edit rather than a quiet one."""
    assert PROBE.TRAVEL_TOLERANCE == 0.05
    assert PROBE.skewed_arms(arms(nothing=8000, a=8000 * 1.04)) == []
    assert PROBE.skewed_arms(arms(nothing=8000, a=8000 * 1.06))


def test_no_control_means_no_comparison_rather_than_a_false_pass():
    """With no `nothing` arm there is nothing to compare against, and inventing a baseline from
    the other arms would compare them to each other under the control's name."""
    assert PROBE.skewed_arms(arms(delete=8000, jump=200)) == []


# ── the wiring ────────────────────────────────────────────────────────
#
# The two functions above are pure and easy to test, which is exactly why they need this section:
# a validator nobody calls passes its own unit tests forever while the probe goes on publishing
# unvalidated arms. These read the source rather than drive a browser, so they run in the same
# CPU job as everything else in this file.

SOURCE = Path(PROBE.__file__).read_text(encoding="utf-8")
MAIN = SOURCE.split("def main()", 1)[1]


def test_every_measured_row_is_validated_before_it_is_kept():
    """`checked_scroll` must be applied to the row inside the repetition loop.

    Placed against `rows[-1]`, the row `run_action` just returned, so it runs before the row can
    reach `med()`.
    """
    assert "checked_scroll(name, rows[-1])" in MAIN, (
        "the measured scroll is appended without validation, so an arm whose scroll did not "
        "complete is still aggregated and published"
    )


def test_the_cross_arm_travel_check_decides_the_exit_code():
    """`skewed_arms` has to be consulted in the block that returns 1, not merely computed."""
    tail = MAIN.split("skewed_arms(", 1)
    assert len(tail) == 2, "main() never calls skewed_arms, so the travel check is dead code"
    assert "return 1" in tail[1], (
        "skewed_arms is called but its result does not reach the exit code, so a run whose arms "
        "scrolled different distances still exits 0 and gets quoted as a result"
    )


def test_the_travel_the_arms_are_compared_on_is_published():
    """`scrolled_px` in the results JSON is what makes the check auditable after the fact."""
    assert '"scrolled_px": med("scrolledPx")' in MAIN
    assert '"scrolled_px_all"' in MAIN
