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
    problems = PROBE.scroll_row_problems(good_row(ran = False))
    assert problems, "a scroll that never ran was accepted"
    assert "null" in problems[0]


def test_a_scroll_that_never_went_quiet_is_rejected():
    """`__hv.quiet()` returns null when the settle timeout expires under continuous activity, so
    settle_ms would be a median of the repetitions that happened to settle."""
    problems = PROBE.scroll_row_problems(good_row(settleMs = None))
    assert any("never went quiet" in p for p in problems), problems


def test_a_scroll_with_no_timing_is_rejected():
    problems = PROBE.scroll_row_problems(good_row(gestureMs = None))
    assert any("no gesture duration" in p for p in problems), problems


def test_a_viewport_that_did_not_move_is_rejected():
    """Zero travel is the failure that looks most like a result: every timing field is present and
    numeric, and the row reads as a very fast scroll."""
    problems = PROBE.scroll_row_problems(good_row(scrolledPx = 0))
    assert any("travelled only 0px" in p for p in problems), problems


def test_a_partially_completed_scroll_is_rejected():
    """The case a bare `> 0` check cannot see. A viewport clamped or snapped back for part of a
    repetition still reports a positive number, and `skewed_arms` compares MEDIANS, so one short
    repetition among complete ones is hidden while its timing stays in the experiment."""
    half = PROBE.hv.REQUESTED_SCROLL_PX // 2
    problems = PROBE.scroll_row_problems(good_row(scrolledPx = half))
    assert any(f"travelled only {half}px" in p for p in problems), problems


def test_a_gesture_a_fraction_short_is_still_accepted():
    """The tolerance is for the last step of a reversal clamping at a boundary. Without this the
    check above could be met by demanding an exact figure no real run produces."""
    almost = int(PROBE.hv.REQUESTED_SCROLL_PX * 0.95)
    assert PROBE.scroll_row_problems(good_row(scrolledPx = almost)) == []


def test_the_probe_and_the_harness_share_one_definition_of_a_completed_gesture():
    """Two copies of this threshold is exactly how the probe came to accept any travel above zero
    while the harness required 90% of the requested distance."""
    source = Path(PROBE.__file__).read_text(encoding = "utf-8")
    assert (
        "hv.scroll_travel_shortfall(" in source
    ), "the probe judges scroll completion on its own terms again"
    assert (
        "hv.jump_anchor_shortfall(" in source
    ), "the probe judges the jump anchor on its own terms again"
    assert PROBE.hv.SCROLL_TRAVEL_TOLERANCE == 0.9
    assert PROBE.hv.JUMP_ANCHOR_TOLERANCE_PX == 2


def test_a_missing_travel_field_is_rejected():
    problems = PROBE.scroll_row_problems(good_row(scrolledPx = None))
    assert any("no travel" in p for p in problems), problems


def test_checked_scroll_raises_so_main_marks_the_arm_failed():
    """`main()` records a raise as a failed arm, keeps it out of the table and exits 1. Returning
    a problem list instead would leave the arm in the comparison."""
    with pytest.raises(RuntimeError) as exc:
        PROBE.checked_scroll("delete", good_row(ran = False))
    assert "delete" in str(exc.value)


# ── the cross-arm comparison ──────────────────────────────────────────


def arms(**travel) -> dict:
    return {n: {"scrolled_px": px} for n, px in travel.items()}


def test_arms_that_scrolled_the_same_distance_are_accepted():
    assert PROBE.skewed_arms(arms(nothing = 8000, delete = 8000, jump = 7900)) == []


def test_an_arm_that_scrolled_a_different_distance_is_rejected():
    """A shorter thread under one predecessor means a shorter gesture, and the difference is then
    reported as that predecessor's cost."""
    skewed = PROBE.skewed_arms(arms(nothing = 8000, delete = 5200))
    assert len(skewed) == 1 and "delete" in skewed[0], skewed


def test_the_tolerance_is_slack_not_a_licence():
    """Pinned so widening it later is a visible edit rather than a quiet one."""
    assert PROBE.TRAVEL_TOLERANCE == 0.05
    assert PROBE.skewed_arms(arms(nothing = 8000, a = 8000 * 1.04)) == []
    assert PROBE.skewed_arms(arms(nothing = 8000, a = 8000 * 1.06))


def test_no_control_means_no_comparison_rather_than_a_false_pass():
    """With no `nothing` arm there is nothing to compare against, and inventing a baseline from
    the other arms would compare them to each other under the control's name."""
    assert PROBE.skewed_arms(arms(delete = 8000, jump = 200)) == []


# ── the wiring ────────────────────────────────────────────────────────
#
# The two functions above are pure and easy to test, which is exactly why they need this section:
# a validator nobody calls passes its own unit tests forever while the probe goes on publishing
# unvalidated arms. These read the source rather than drive a browser, so they run in the same
# CPU job as everything else in this file.

SOURCE = Path(PROBE.__file__).read_text(encoding = "utf-8")
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


# ── travel is observed, not planned ───────────────────────────────────
#
# `scroll_row_problems` above rejects a row that reports no travel, but that is only worth
# anything if the number it reads is what the viewport DID rather than what the harness ASKED
# for. SCROLL_JS accumulated `next - target`, both of which are requests. Studio replaces
# assistant-ui's autoscroll with an intent-aware one that can snap a programmatic scroll straight
# back, and the browser clamps at either end, so a gesture that moved nothing still reported the
# full planned distance and every check downstream passed on it.

HARNESS_SOURCE = (Path(__file__).resolve().parent / "playwright_heavy_thread.py").read_text(
    encoding = "utf-8"
)


def _loop_body(script: str) -> str:
    return script.split("window.__hv.begin();", 1)[1]


def test_the_scroll_gesture_accumulates_what_the_viewport_did() -> None:
    body = _loop_body(PROBE.hv.SCROLL_JS)
    assert (
        "const landed = viewport.scrollTop;" in body
    ), "the scroll no longer reads back where it landed"
    assert "travelled += Math.abs(landed - observed);" in body, (
        "travel is accumulated from the requested position again, so a viewport that was clamped "
        "or snapped back still reports the full planned distance"
    )
    assert "travelled += Math.abs(next - target);" not in body


def test_the_jump_reports_observed_travel_and_where_it_started() -> None:
    jump = PROBE.hv.JUMP_JS
    assert (
        "travelledPx: Math.abs(landedAt - startedFrom)," in jump
    ), "the jump reports its planned full-height travel again rather than what it covered"
    assert "const startedFrom = viewport.scrollTop;" in jump
    assert "travelledPx: bottom," not in jump


# ── every raw jump starts where the harness's does ────────────────────


def test_the_jump_predecessor_applies_the_registered_anchor() -> None:
    """The measured scroll leaves the viewport thousands of px above the bottom, so from
    repetition 2 on an unanchored jump is a shorter gesture aggregated under the same median."""
    source = Path(PROBE.__file__).read_text(encoding = "utf-8")
    body = source.split("def before_jump", 1)[1].split("\ndef ", 1)[0]
    assert (
        'hv.ACTION_SETUPS["jump"]' in body
    ), "before_jump calls JUMP_JS raw, bypassing the anchor the harness applies to its own jump"
    assert body.index('ACTION_SETUPS["jump"]') < body.index(
        "hv.JUMP_JS"
    ), "the anchor runs after the jump, which is not an anchor"


def jump_out(**overrides) -> dict:
    out = {"landedAt": 0, "travelledPx": 8000, "startedFrom": 8000, "bottom": 8000}
    out.update(overrides)
    return out


def test_a_jump_from_the_bottom_is_accepted() -> None:
    """The control. Without it the two checks below could be met by rejecting every jump."""
    assert PROBE.PREDECESSOR_PROOFS["jump"](jump_out()) == []


def test_a_jump_that_began_part_way_up_the_thread_is_rejected() -> None:
    problems = PROBE.PREDECESSOR_PROOFS["jump"](jump_out(startedFrom = 3000, travelledPx = 3000))
    assert any("above the bottom" in p for p in problems), problems


def test_a_jump_that_did_not_say_where_it_started_is_rejected() -> None:
    """Reporting nothing must not read as reporting something acceptable."""
    out = jump_out()
    del out["startedFrom"]
    problems = PROBE.PREDECESSOR_PROOFS["jump"](out)
    assert any("unverifiable" in p for p in problems), problems


def test_a_jump_that_did_not_reach_the_bottom_is_still_rejected() -> None:
    """The pre-existing half of the proof, kept red-able beside the new half."""
    problems = PROBE.PREDECESSOR_PROOFS["jump"](jump_out(landedAt = 4000))
    assert any("landed at" in p for p in problems), problems


# ── the boundary counter counts the gesture, not the anchor ───────────


def test_the_boundary_counter_is_armed_after_the_anchor() -> None:
    """`ACTION_SETUPS` anchors the viewport to the bottom before the gesture, and after the
    predecessor or the previous repetition that is a full-height reposition which fires its own
    pointerover/pointerout. Armed beforehand, those landed in the measured gesture's count by an
    amount that depended on where each arm's predecessor had left the viewport, so the counter
    would appear to support a predecessor effect generated outside the gesture being compared.
    """
    body = SOURCE.split("install_boundary_counter = ", 1)[1]
    assert (
        "after_setup = lambda p: p.evaluate(install_boundary_counter)" in body
    ), "the boundary counter is not armed from run_action's after_setup hook"
    # The bare pre-arm has to be GONE, not merely joined by the hook, or both run and the counter
    # is installed twice with the first one still spanning the anchor.
    stray = [
        line
        for line in body.splitlines()
        if "page.evaluate(install_boundary_counter)" in line and "after_setup" not in line
    ]
    assert not stray, f"the counter is still armed before the anchor as well: {stray}"
