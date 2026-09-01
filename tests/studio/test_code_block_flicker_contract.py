# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the code-block flicker harness is held to.

`tests/studio/playwright_code_block_flicker.py` decides whether a code block flickered from a
per-frame log of rendered heights. Two failures there never show up as a failing run: the detector
never fires and every tree looks clean, or it fires on a height change that is not a flicker and
every tree looks broken.

So it is exercised here against hand-written frame logs, including the ones it must NOT report,
and the harness is held to driving at least one variant known to flicker, so a run reporting "no
flicker" has said something.

The analysis lives in `_code_block_flicker_analysis.py` so this file can import it wherever the
CPU suite runs: the harness imports playwright, this does not.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"
FRONTEND = ROOT / "studio" / "frontend"
sys.path.insert(0, str(STUDIO_TESTS))

from _code_block_flicker_analysis import (  # noqa: E402
    PLACEHOLDER_HI,
    PLACEHOLDER_LO,
    RECOVERY_FRAMES,
    SHIFT_PX,
    TALL_PX,
    analyse_stream,
    analyse_sweep,
)

HARNESS = STUDIO_TESTS / "playwright_code_block_flicker.py"
FIXTURE = FRONTEND / "smoke-code-block-flicker-main.tsx"
PAGE = FRONTEND / "smoke-code-block-flicker.html"


def frame(
    heights: list[float],
    scroll_height: float | None = None,
    **overrides,
) -> dict:
    """One sampled frame. Defaults are a still thread: nothing scrolls, nothing moves."""
    base = {
        "t": 0.0,
        "heights": list(heights),
        "tops": [sum(heights[:i]) for i in range(len(heights))],
        "scrollTop": 0.0,
        "scrollHeight": scroll_height if scroll_height is not None else sum(heights),
        "clientHeight": 900.0,
        "anchorTop": 0.0,
        "running": False,
    }
    base.update(overrides)
    return base


# the detector fires on the thing it is for


def test_a_one_frame_collapse_to_the_placeholder_is_a_collapse() -> None:
    """The measured shape: 1722px, one frame at 226px, back to 1722px."""
    frames = [frame([1722.0])] * 3 + [frame([226.0])] + [frame([1722.0])] * 3
    result = analyse_stream(frames)
    assert result["collapses"] == 1
    assert result["placeholderFrames"] == 1
    assert result["worstDropPx"] == pytest.approx(1496.0)
    assert result["detail"][0]["heightAtFloor"] == 226.0
    assert result["detail"][0]["frames"] == 1


def test_a_collapse_that_deepens_reports_its_deepest_point() -> None:
    """A drop arriving over several frames is one collapse, measured at the floor.

    The reported drop must agree with the recorded floor, or a red run understates what it caught:
    1700 -> 700 -> 200 is a 1500px collapse, not a 1000px one.
    """
    frames = [frame([1700.0]), frame([700.0]), frame([200.0]), frame([1700.0])]
    result = analyse_stream(frames)
    assert result["collapses"] == 1
    assert result["worstDropPx"] == pytest.approx(1500.0)
    assert result["detail"][0]["heightAtFloor"] == 200.0


def test_each_block_is_counted_separately() -> None:
    """Three fences in one reply collapse one after another, and that is three, not one."""
    tall = [1700.0, 1700.0, 1700.0]
    frames = [frame(tall)] * 2
    for index in range(3):
        collapsed = list(tall)
        collapsed[index] = 226.0
        frames += [frame(collapsed), frame(tall)]
    assert analyse_stream(frames)["collapses"] == 3


def test_a_collapse_below_the_placeholder_band_still_counts() -> None:
    """`contain-intrinsic-size: auto none` collapses toward zero, not to 200px: a different
    fallback, not a different bug, so it counts but claims no placeholder attribution."""
    frames = [frame([1700.0])] * 2 + [frame([2.0])] + [frame([1700.0])] * 2
    result = analyse_stream(frames)
    assert result["collapses"] == 1
    assert result["placeholderFrames"] == 0


def test_a_collapsing_block_takes_the_scroll_height_with_it() -> None:
    frames = (
        [frame([1700.0, 1700.0], scroll_height = 3400.0)] * 2
        + [frame([1700.0, 226.0], scroll_height = 1926.0)]
        + [frame([1700.0, 1700.0], scroll_height = 3400.0)] * 2
    )
    assert analyse_stream(frames)["scrollHeightDips"] == 1


# and not on things that are not it


def test_a_still_thread_reports_nothing() -> None:
    result = analyse_stream([frame([1700.0, 1700.0])] * 40)
    assert result["collapses"] == 0
    assert result["placeholderFrames"] == 0
    assert result["scrollHeightDips"] == 0
    assert result["anchorShiftPx"] == 0.0


def test_a_block_that_goes_short_and_stays_short_is_not_a_flicker() -> None:
    """A drop with no recovery is a different bug, and reporting it here would hide this one. It
    is still recorded, with a null recovery frame, so it cannot be silently dropped."""
    frames = [frame([1700.0])] * 3 + [frame([226.0])] * (RECOVERY_FRAMES + 5)
    result = analyse_stream(frames)
    assert result["collapses"] == 0
    assert result["detail"], "a drop that never recovered must still be recorded"
    assert result["detail"][0]["toFrame"] is None


def test_a_drop_still_open_when_the_log_ends_is_recorded() -> None:
    """The tail is shorter than RECOVERY_FRAMES, so the realistic case ends mid-drop.

    2500ms is ~150 frames at 60Hz against RECOVERY_FRAMES of 240, so a block collapsing at
    finalization and staying short never trips the threshold. It must still appear in `detail`,
    the one place a non-recovering drop is promised to show up.
    """
    frames = [frame([1700.0])] * 3 + [frame([226.0])] * 150
    result = analyse_stream(frames)
    assert result["collapses"] == 0, "it never came back, so it is not a flicker"
    assert result["detail"], "a drop open at the end of the log must still be recorded"
    assert result["detail"][0]["toFrame"] is None
    assert result["detail"][0]["heightAtFloor"] == 226.0


def test_a_block_growing_as_its_content_arrives_is_not_a_collapse() -> None:
    """A fence gets taller line by line while it streams. Nothing here is a drop."""
    frames = [frame([200.0 * (i + 1)]) for i in range(12)]
    assert analyse_stream(frames)["collapses"] == 0


def test_a_short_block_is_never_read_as_a_collapsed_tall_one() -> None:
    """A two-line fence really is 120px. Without the floor it would look like a placeholder."""
    assert TALL_PX > PLACEHOLDER_HI
    frames = [frame([TALL_PX - 40.0])] * 2 + [frame([120.0])] + [frame([TALL_PX - 40.0])] * 2
    assert analyse_stream(frames)["collapses"] == 0


def test_a_thread_that_simply_gets_shorter_is_not_a_dip() -> None:
    """Deleting a message shortens the column and it stays short. That is not a flicker."""
    frames = [frame([1700.0, 1700.0], scroll_height = 3400.0)] * 3 + [
        frame([1700.0], scroll_height = 1700.0)
    ] * 20
    assert analyse_stream(frames)["scrollHeightDips"] == 0


def test_blocks_appearing_and_disappearing_are_not_collapses() -> None:
    """The heights array is not a fixed-width record: it grows as blocks are APPENDED, and empties
    when the thread unmounts. An absent block has no height, which is not a height of zero:
    reading it as zero turns every teardown into a column of collapses.
    """
    appearing = [frame([1700.0])] * 3 + [frame([1700.0, 900.0])] * 3
    assert analyse_stream(appearing)["collapses"] == 0

    torn_down = [frame([1700.0, 1700.0])] * 3 + [frame([])] * 3 + [frame([1700.0, 1700.0])] * 3
    assert analyse_stream(torn_down)["collapses"] == 0


def test_scrolling_does_not_move_the_anchor() -> None:
    """The anchor is read in document space, so a scroll of 400px a frame is not a shift."""
    frames = [frame([1700.0], anchorTop = -400.0 * i, scrollTop = 400.0 * i) for i in range(10)]
    assert analyse_stream(frames)["anchorShiftPx"] == 0.0


def test_content_relaid_out_above_the_anchor_is_a_shift() -> None:
    frames = [
        frame([1700.0], anchorTop = 0.0, scrollTop = 0.0),
        frame([1700.0], anchorTop = -900.0, scrollTop = 0.0),
    ]
    assert analyse_stream(frames)["anchorShiftPx"] == 900.0


def test_a_sweep_over_a_thread_that_knows_its_own_size_reports_no_shift() -> None:
    tops = [0.0, 1700.0, 3400.0]
    frames = [{"tops": tops, "scrollHeight": 5100.0, "heights": [1700.0] * 3} for _ in range(20)]
    result = analyse_sweep(frames)
    assert result["shiftFrames"] == 0
    assert result["worstShiftPx"] == 0.0
    assert result["scrollHeightGrowthPx"] == 0


def test_a_block_expanding_as_it_is_reached_moves_everything_below_it() -> None:
    """The placeholder cost, in the phase where the user sees it."""
    before = {"tops": [0.0, 226.0, 452.0], "scrollHeight": 678.0, "heights": [226.0] * 3}
    after = {"tops": [0.0, 1700.0, 1926.0], "scrollHeight": 2152.0, "heights": [1700.0] * 3}
    result = analyse_sweep([before, before, after, after])
    assert result["shiftFrames"] == 1
    assert result["worstShiftPx"] == pytest.approx(1474.0)
    assert result["scrollHeightGrowthPx"] == pytest.approx(1474.0)


def test_sub_pixel_wobble_is_not_a_shift() -> None:
    """Fractional layout differences are not the page moving under the user."""
    assert SHIFT_PX >= 1
    a = {"tops": [0.0, 1700.0], "scrollHeight": 3400.0, "heights": [1700.0, 1700.0]}
    b = {"tops": [0.0, 1700.4], "scrollHeight": 3400.4, "heights": [1700.0, 1700.4]}
    assert analyse_sweep([a, b])["shiftFrames"] == 0


def harness_source() -> str:
    return HARNESS.read_text(encoding = "utf-8")


def module_assignment(source: str, name: str) -> ast.expr | None:
    """The value assigned to a module-level `name`, or None.

    Parsed rather than grepped: a substring check passes on a constant renamed to `NAME_DISABLED`
    and left unread, which is how one of these guards gets turned off without a test noticing.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def test_the_run_drives_a_variant_that_is_required_to_flicker() -> None:
    """Without one, "no flicker" cannot be told apart from "measured nothing"."""
    source = harness_source()
    must = module_assignment(source, "MUST_FLICKER")
    must_not = module_assignment(source, "MUST_NOT_FLICKER")
    assert must is not None, "the harness declares no variant that has to flicker"
    assert must_not is not None, "the harness declares no variant that must not flicker"
    assert isinstance(must, ast.Set) and must.elts, "MUST_FLICKER is empty"
    assert isinstance(must_not, ast.Set) and must_not.elts, "MUST_NOT_FLICKER is empty"
    # The exit code has to rest on both halves, not just on the tree behaving.
    verdict = source[source.index("if variant in MUST_FLICKER") :]
    assert "failures.append" in verdict


def test_a_filtered_variant_set_cannot_drop_every_positive_control() -> None:
    """`SMOKE_FLICKER_VARIANTS=tree` would otherwise pass while proving nothing. The verdict loop
    only judges variants it ran, so a set keeping no MUST_FLICKER member is refused up front."""
    source = harness_source()
    assert (
        "MUST_FLICKER & set(VARIANTS)" in source
    ), "the harness does not check that the selected variants keep a positive control"
    guard = source[source.index("if not MUST_FLICKER & set(VARIANTS)") :]
    assert "raise SystemExit" in guard[:400], "the missing-control check does not stop the run"


def test_the_fixture_offers_the_state_the_override_was_added_for() -> None:
    """The pre-override stylesheet has to be reachable, or the flicker cannot be reproduced."""
    fixture = FIXTURE.read_text(encoding = "utf-8")
    assert "streamdown:" in fixture
    assert "content-visibility: auto" in fixture
    assert "contain-intrinsic-size: auto 200px" in fixture


def test_the_variant_stylesheets_are_checked_to_have_won_the_cascade() -> None:
    """A variant that loses to the tree measures the tree under another name. That happened: every
    variant computed identically, all reported zero collapses, and the run read as "nothing to
    fix"."""
    source = harness_source()
    settled = module_assignment(source, "EXPECTED_COMPUTED")
    running = module_assignment(source, "EXPECTED_COMPUTED_RUNNING")
    assert settled is not None and getattr(settled, "keys", None), "no settled cascade check"
    assert running is not None and getattr(
        running, "keys", None
    ), "a settled-state cascade check alone passes on a variant that only loses while streaming"
    assert (
        "EXPECTED_COMPUTED_RUNNING.get(variant" in source
    ), "the mid-stream expectation is declared but never compared against anything"


def test_the_page_and_its_entry_agree() -> None:
    assert FIXTURE.name in PAGE.read_text(encoding = "utf-8")


def test_the_placeholder_band_matches_the_fallback_the_library_sets() -> None:
    """200px inline, plus the wrapper's padding and its header row."""
    assert PLACEHOLDER_LO < 200 < PLACEHOLDER_HI
