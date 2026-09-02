# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE SCROLL GESTURE, bounded at both ends and against the extent's own allowance.

`scroll_travelled` is the only behavioural invariant with two independent quantities in it -- how
far the gesture went, and whether the two arms agree about how far it COULD have gone -- and it had
no direct test of its own. Both of its defects were the same mistake in opposite directions: one
quantity judged against a tolerance meant for a different quantity, and one judged against no upper
bound at all.

Kept out of the windowed-parity module deliberately. That module is the ui-parity command's own
suite and these are `analysis/behaviour.py`'s, so they live beside the code they pin.
"""

from __future__ import annotations

import sys
from pathlib import Path

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.analysis import behaviour as B  # noqa: E402
from studiobench.analysis import parity as P  # noqa: E402


def _capture(
    mounted: int,
    total: int,
    digest: str = "d",
) -> dict:
    """A parity capture in the shape scene/parity.js writes one; `mounted < total` is a window."""
    return {
        "parity_attempted": True,
        "root_kind": "thread",
        "digest": digest,
        "chars": 100,
        "messages": [
            {"i": i, "role": "assistant", "digest": f"{digest}{i}", "chars": 10}
            for i in range(mounted)
        ],
        "overlays": [],
        "styles": {"elements": mounted, "digest": "s", "capped": False},
        "mounted_messages": mounted,
        "thread_total": total,
    }


def _row(action: str, capture: dict, **expect) -> dict:
    return {
        "row_type": "action",
        "action": action,
        "ran": True,
        "expect_ok": True,
        "expect": dict(expect),
        "timings": {},
        "parity": capture,
        "census": {"viewport_scroll_height": 10_000},
    }


# ── the scroll gesture is bounded at both ends, at the extent's own allowance ─────


def _scroll_row(
    mounted,
    *,
    fraction = 1.0,
    bottom = 9_200,
    commanded = 5_880,
    client = None,
):
    row = _row(
        "scroll_after",
        _capture(mounted, 18),
        travel_fraction = fraction,
        travelled_px = round(fraction * commanded),
        commanded_px = commanded,
        bottom = bottom,
    )
    if client is not None:
        row["census"]["viewport_client_height"] = client
    return row


def test_an_extent_inside_the_declared_allowance_is_not_reported_broken():
    """THE FALSE RED. `scroll_extent` grants the extent 10% because a windowed list computes its
    total height from estimated row heights, and says so in its own comment. `scroll_travelled`
    then compared the SAME physical quantity through `_same_number`, which routes through the 2%
    `EXACT_TOLERANCE`, so an arm inside the declared allowance was reported behaviourally BROKEN.

    A false red is not free here: a broken behavioural invariant removes the cell from
    `readings_by_arm` and takes its healthy partner with it through the arm intersection, and
    `unmeasured_planned_cells` now VOIDS the plan over the hole that leaves."""
    base = _scroll_row(18, bottom = 9_200, client = 800)
    treat = _scroll_row(6, bottom = 8_600, client = 800)
    treat["census"]["viewport_scroll_height"] = 9_400  # 6% out: inside EXTENT_TOLERANCE
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.MATCH, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["scroll_bottom_agrees"]["ok"] is True, checks["scroll_bottom_agrees"]
    assert checks["scroll_extent"]["ok"] is True, checks["scroll_extent"]


def test_an_extent_outside_the_declared_allowance_is_still_reported_broken():
    """THE CONTROL. Widening this check must not stop it catching an extent that is a FRACTION of
    the real one, which is the failure `EXTENT_TOLERANCE` was chosen against: a virtualizer that
    drops rows instead of sizing spacers has a scrollbar that lies."""
    base = _scroll_row(18, bottom = 9_200, client = 800)
    treat = _scroll_row(6, bottom = 4_200, client = 800)
    treat["census"]["viewport_scroll_height"] = 5_000
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN, got
    assert "scroll_bottom_agrees" in got["reason"], got


def test_the_extent_is_reconstructed_so_both_checks_answer_about_one_scrollbar():
    """`bottom` is `scrollHeight - clientHeight` and `_drift` is PROPORTIONAL, so subtracting a
    shared viewport height amplifies the drift by `H / (H - C)`. Left as a comparison of `bottom`,
    the same tolerance means something tighter than it says and the two checks print two different
    percentages for one scrollbar. The census has carried `viewport_client_height` beside the
    scroll height all along."""
    base = _scroll_row(18, bottom = 9_200, client = 800)
    treat = _scroll_row(6, bottom = 8_600, client = 800)
    treat["census"]["viewport_scroll_height"] = 9_400
    checks = {c["invariant"]: c for c in B.compare_behaviour(base, treat)["checks"]}
    # 10000 vs 9400 either way round, so the two details agree to the printed digit.
    assert "6.0% drift" in checks["scroll_bottom_agrees"]["detail"], checks
    assert "6.0% drift" in checks["scroll_extent"]["detail"], checks
    assert "scroll extent" in checks["scroll_bottom_agrees"]["detail"], checks


def test_a_payload_with_no_client_height_falls_back_and_says_so():
    """A row from a checkout that predates the field, or one whose census failed. It compares
    `bottom` as before rather than refusing, and the detail names which of the two it compared so
    the tighter effective allowance is not silent."""
    base = _scroll_row(18, bottom = 9_200)
    treat = _scroll_row(6, bottom = 9_100)
    checks = {c["invariant"]: c for c in B.compare_behaviour(base, treat)["checks"]}
    assert "no client height on both arms" in checks["scroll_bottom_agrees"]["detail"], checks
    assert checks["scroll_bottom_agrees"]["ok"] is True, checks


def test_one_arm_missing_its_client_height_does_not_compare_an_extent_with_a_bottom():
    """A CENSUS THAT FAILED ON ONE ARM, or two payloads of different vintages. Reconstructing per
    arm and comparing whatever each produced put a `scrollHeight` beside a `bottom`: identical
    viewports at `bottom` 1,200 over a client height of 800 came out 2,000 against 1,200 and
    BROKEN at 40% drift, under a detail line that said `no client height on both arms`. The
    decision belongs to the pair, so both arms are reconstructed or neither is."""
    base = _scroll_row(18, bottom = 1_200, client = 800)
    treat = _scroll_row(18, bottom = 1_200)
    for row in (base, treat):
        row["census"]["viewport_scroll_height"] = 2_000
    got = B.compare_behaviour(base, treat)
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["scroll_bottom_agrees"]["ok"] is True, checks
    assert "1200.0 vs 1200.0" in checks["scroll_bottom_agrees"]["detail"], checks
    assert got["verdict"] == P.MATCH, got


def test_client_heights_that_disagree_are_not_treated_as_a_shared_offset():
    """`clientHeight` is a shared offset only when it is shared. Two arms reporting the same
    10,000px `scrollHeight` at client heights of 800 and 2,000 have 9,200px and 8,000px of room
    for the gesture, and adding each arm's own viewport back reported MATCH at 0.0% drift over
    that 1,200px difference. `scroll_extent` already compares the scroll heights; this check is
    about the range the gesture actually had, so it falls back to the raw bottoms and says why."""
    base = _scroll_row(18, bottom = 9_200, client = 800)
    treat = _scroll_row(18, bottom = 8_000, client = 2_000)
    got = B.compare_behaviour(base, treat)
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["scroll_bottom_agrees"]["ok"] is False, checks
    assert "client heights differ: 800.0 vs 2000.0" in checks["scroll_bottom_agrees"]["detail"]
    assert checks["scroll_extent"]["ok"] is True, checks
    assert got["verdict"] == B.BROKEN, got


def test_client_heights_that_agree_are_still_treated_as_a_shared_offset():
    """The positive control for the two above: the ordinary case, where both arms publish the same
    viewport height, must still be compared as extents rather than pushed onto the fallback."""
    base = _scroll_row(18, bottom = 9_200, client = 800)
    treat = _scroll_row(6, bottom = 8_600, client = 800)
    treat["census"]["viewport_scroll_height"] = 9_400
    checks = {c["invariant"]: c for c in B.compare_behaviour(base, treat)["checks"]}
    assert "scroll extent 10000.0 vs 9400.0" in checks["scroll_bottom_agrees"]["detail"], checks
    assert checks["scroll_bottom_agrees"]["ok"] is True, checks


def test_a_gesture_that_overshot_its_command_is_reported():
    """THE UNBOUNDED SIDE. The predicate was `fraction >= 0.9` and nothing above it, so a treatment
    arm whose viewport moved TWICE as far as the gesture asked passed exactly as 1.0 did and the
    pair returned MATCH. `travelled` is a sum of `|scrollTop_after - target_before|`, so every
    pixel above `commanded` is the viewport being moved by something other than the gesture --
    which is the anchor instability this action exists to detect, in the other direction."""
    base = _scroll_row(18, fraction = 1.0, bottom = 9_200, client = 800)
    treat = _scroll_row(6, fraction = 2.0, bottom = 9_200, client = 800)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN, got
    assert "scroll_travelled:treatment" in got["reason"], got


def test_an_ordinary_estimate_correction_still_passes_the_ceiling():
    """THE CONTROL, and the reason the ceiling is DERIVED rather than picked to look symmetric with
    0.9. A windowed list correcting estimated row heights moves the offset by the error in the
    estimate, and this file already declares how far the extent may be wrong. So the allowance is
    `EXTENT_TOLERANCE` of the PAIR'S reference extent -- 1,000px of a 10,000px extent against a
    5,880px gesture, a ceiling of 1.170 -- and a correction inside it is ordinary rather than a
    finding."""
    base = _scroll_row(18, fraction = 1.0, bottom = 9_200, client = 800)
    treat = _scroll_row(6, fraction = 1.1, bottom = 9_200, client = 800)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.MATCH, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert "allowed 0.9 to 1.170" in checks["scroll_travelled:treatment"]["detail"], checks


def test_the_ceiling_grants_the_tolerance_against_the_extent_not_against_bottom():
    """THE SAME MISTAKE THE CHECK BELOW WAS FIXED FOR, in the ceiling. `bottom` is
    `scrollHeight - clientHeight`, so `EXTENT_TOLERANCE` taken on it grants 920px of a 10,000px
    extent behind an 800px viewport where the tolerance says 1,000. A 941px correction is 9.4% of
    the extent -- inside the declared 10% -- and came out BROKEN.

    Pinned at both edges so the ceiling is a bound and not merely a larger number: 1.160 is inside
    the allowance and 1.171 is outside it."""
    base = _scroll_row(18, fraction = 1.0, bottom = 9_200, client = 800)
    inside = _scroll_row(6, fraction = 1.16, bottom = 9_200, client = 800)
    got = B.compare_behaviour(base, inside)
    assert got["verdict"] == P.MATCH, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert (
        "of the pair's 10000.0 px reference extent"
        in checks["scroll_travelled:treatment"]["detail"]
    )

    outside = _scroll_row(6, fraction = 1.171, bottom = 9_200, client = 800)
    beyond = B.compare_behaviour(base, outside)
    assert beyond["verdict"] == B.BROKEN, beyond
    assert "scroll_travelled:treatment" in beyond["reason"], beyond


def test_the_ceiling_and_the_extent_check_enforce_one_tolerance_on_one_quantity():
    """TWO DENOMINATORS FOR ONE TOLERANCE, and the pair disagreed inside it.

    `scroll_extent` scores drift through `_drift`, which divides by the LARGER of the two extents.
    The ceiling divided by the arm's OWN. So extents of 10,000 and 9,050 pass `scroll_extent` at
    9.5% drift, while the 950px correction that closes that very gap was BROKEN against a ceiling
    granting 10% of 9,050 -- 1.162 travelled against 1.154 allowed. The arm that most needs the
    allowance is exactly the arm whose own extent is the worse yardstick for it.

    A false red is not free: it removes the cell from `readings_by_arm`, takes its healthy partner
    with it through the arm intersection, and `unmeasured_planned_cells` can then VOID the plan.

    Pinned at both edges, so this is a bound and not merely a bigger number: the reference ceiling
    is 1.170, and a gesture past it is still reported."""
    base = _scroll_row(18, fraction = 1.0, bottom = 9_200, client = 800)
    treat = _scroll_row(6, fraction = (5_880 + 950) / 5_880, bottom = 8_250, client = 800)
    treat["census"]["viewport_scroll_height"] = 9_050

    got = B.compare_behaviour(base, treat)
    checks = {c["invariant"]: c for c in got["checks"]}
    # The pair really is inside the extent allowance; that is what makes the red false.
    assert checks["scroll_extent"]["ok"] is True, checks
    assert "9.5% drift, 10% allowed" in checks["scroll_extent"]["detail"], checks
    assert checks["scroll_travelled:treatment"]["ok"] is True, checks
    assert "allowed 0.9 to 1.170" in checks["scroll_travelled:treatment"]["detail"], checks
    assert (
        "of the pair's 10000.0 px reference extent"
        in (checks["scroll_travelled:treatment"]["detail"])
    ), checks

    # Still a ceiling: past the pair's reference allowance it is reported.
    beyond = _scroll_row(6, fraction = (5_880 + 1_060) / 5_880, bottom = 8_250, client = 800)
    beyond["census"]["viewport_scroll_height"] = 9_050
    worse = B.compare_behaviour(base, beyond)
    assert worse["verdict"] == B.BROKEN, worse
    assert "scroll_travelled:treatment" in worse["reason"], worse


def test_an_arm_with_no_extent_does_not_borrow_its_partners_ceiling():
    """The degradation stays a degradation. Bounding an arm this check has always left unbounded
    above is the one direction that could INVENT a red rather than retire one, so an arm carrying
    no extent of its own still gets the lower bound alone and says so."""
    base = _scroll_row(18, fraction = 1.0, bottom = 9_200, client = 800)
    treat = _scroll_row(6, fraction = 2.0, bottom = None, client = 800)
    treat["census"]["viewport_scroll_height"] = None

    checks = {c["invariant"]: c for c in B.compare_behaviour(base, treat)["checks"]}
    assert checks["scroll_travelled:treatment"]["ok"] is True, checks
    assert "NO CEILING" in checks["scroll_travelled:treatment"]["detail"], checks


def test_the_lower_bound_still_catches_a_gesture_that_was_snapped_back():
    """The bound this invariant was written for: Unsloth's intent-aware autoscroll snapping a
    programmatic move straight back to the bottom left the gesture having covered nothing."""
    base = _scroll_row(18, fraction = 1.0, bottom = 9_200, client = 800)
    treat = _scroll_row(6, fraction = 0.1, bottom = 9_200, client = 800)
    assert B.compare_behaviour(base, treat)["verdict"] == B.BROKEN


def test_a_row_with_no_extent_gets_the_lower_bound_and_no_ceiling():
    """Degrades to no ceiling rather than to a guess. A ceiling of zero would fail every correct
    arm; the row says which of the two it applied."""
    base = _scroll_row(18, fraction = 1.0, bottom = 9_200, client = 800)
    treat = _scroll_row(6, fraction = 2.0, bottom = 9_200, client = 800)
    del treat["expect"]["bottom"]
    del base["expect"]["bottom"]
    checks = {c["invariant"]: c for c in B.compare_behaviour(base, treat)["checks"]}
    assert checks["scroll_travelled:treatment"]["ok"] is True, checks
    assert "NO CEILING" in checks["scroll_travelled:treatment"]["detail"], checks
