# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shows the readiness gate admitting windowed arms and refusing each unready thread, in Chromium."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.runtime.readiness import (  # noqa: E402
    COVERAGE_COMPLETE,
    COVERAGE_INCOMPLETE,
    COVERAGE_NOT_APPLICABLE,
    COVERAGE_UNMEASURED,
    MODE_FULL,
    MODE_WINDOWED,
    ThreadNotReady,
    evaluate,
    ordinal_coverage,
    probe_thread_completeness,
    wait_for_thread_ready,
)
from studiobench.runtime.seeder import turn_marker  # noqa: E402

TURNS = 9
MESSAGES = TURNS * 2
WINDOW = 6
# Completeness tests step by two rows; non-overlapping stops can only report NOT MEASURED.
ROW_PX = 120

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"
_FIXTURE_JS = _HERE.parent / "thread_fixture.js"


def _skip_reason() -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"playwright is not installed: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason = _skip_reason() or "")


@pytest.fixture(scope = "module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args = ["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium could not be launched: {exc}")
        yield b
        b.close()


def _page(
    browser,
    mode: str,
    turns: int = TURNS,
):
    page = browser.new_page(viewport = {"width": 900, "height": 600})
    page.set_content("<!doctype html><meta charset=utf-8><body></body>")
    page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
    page.add_script_tag(content = _FIXTURE_JS.read_text(encoding = "utf-8"))
    built = page.evaluate(
        "(o) => window.__fixture.build(o)",
        {"mode": mode, "turns": turns, "windowSize": WINDOW},
    )
    assert built["total"] == turns * 2
    return page


def _lines() -> tuple[list[str], callable]:
    got: list[str] = []
    return got, got.append


def test_the_fixture_marker_matches_the_seeder_exactly(browser):
    """The fixture marker must equal the seeder's, or negative cases pass for free and positives fail."""
    page = _page(browser, "full")
    try:
        assert page.evaluate("(i) => window.__fixture.marker(i)", 0) == turn_marker(0, 0)
        assert page.evaluate("(i) => window.__fixture.marker(i)", 8) == turn_marker(8, 8)
    finally:
        page.close()


def test_the_fixture_really_mounts_what_each_mode_claims(browser):
    """The fixture is the instrument here, so its own readings are checked first."""
    expected = {
        "full": MESSAGES,
        "windowed": WINDOW,
        "windowed_no_total": WINDOW,
        "windowed_at_top": WINDOW,
        "windowed_lost_head": WINDOW,
        "windowed_lost_middle": WINDOW,
        "windowed_zero_ordinals": WINDOW,
        "windowed_duplicate_ordinals": WINDOW,
        "windowed_from_one": WINDOW,
    }
    for mode, want in expected.items():
        page = _page(browser, mode)
        try:
            got = page.evaluate("() => window.__sb.dom.messageCount()")
            assert got == want, f"{mode} mounted {got}, expected {want}"
        finally:
            page.close()


def test_thread_total_reads_the_published_setsize_and_falls_back_to_the_count(browser):
    """`threadTotal()` is what every before/after assertion in actions.py now uses."""
    page = _page(browser, "full")
    try:
        assert page.evaluate("() => window.__sb.dom.threadTotal()") == MESSAGES
        assert page.evaluate("() => window.__sb.dom.isWindowed()") is False
    finally:
        page.close()
    page = _page(browser, "windowed")
    try:
        assert page.evaluate("() => window.__sb.dom.threadTotal()") == MESSAGES
        assert page.evaluate("() => window.__sb.dom.messageCount()") == WINDOW
        assert page.evaluate("() => window.__sb.dom.isWindowed()") is True
    finally:
        page.close()


def test_full_mount_is_admitted_in_full_mode(browser):
    page = _page(browser, "full")
    got, log = _lines()
    try:
        r = wait_for_thread_ready(
            page,
            MESSAGES,
            marker = turn_marker(TURNS - 1, TURNS - 1),
            mode = MODE_FULL,
            timeout_s = 20,
            log = log,
        )
    finally:
        page.close()
    assert r.ready
    assert r.conditions["all_messages_mounted"] is True
    assert r.conditions["end_present"] is True
    assert r.conditions["settled"] is True
    assert r.probe["mounted"] == MESSAGES
    # Unsloth publishes no aria-posinset, so ordinal conditions are None (not measured), not a pass.
    assert r.conditions["posinset_ordinals_valid"] is None
    assert r.conditions["posinset_reaches_end"] is None
    assert r.probe["posinset_count"] == 0


def test_a_virtualised_thread_is_admitted_in_windowed_mode(browser):
    """THE WHOLE POINT. Six of eighteen mounted, and the gate says ready."""
    page = _page(browser, "windowed")
    got, log = _lines()
    try:
        r = wait_for_thread_ready(
            page,
            MESSAGES,
            marker = turn_marker(TURNS - 1, TURNS - 1),
            mode = MODE_WINDOWED,
            timeout_s = 20,
            log = log,
        )
    finally:
        page.close()
    assert r.ready, r.reason
    assert r.probe["mounted"] == WINDOW < MESSAGES
    assert r.conditions["total_matches_seeded"] is True
    assert r.conditions["posinset_on_every_row"] is True
    assert r.conditions["posinset_ordinals_valid"] is True
    assert r.conditions["posinset_reaches_end"] is True
    assert r.conditions["anchored_at_end"] is True
    assert r.conditions["end_present"] is True
    assert (r.probe["min_posinset"], r.probe["max_posinset"]) == (MESSAGES - WINDOW + 1, MESSAGES)
    assert r.probe["posinset_distinct"] == WINDOW


@pytest.mark.parametrize("mode", ["windowed", "windowed_flat"])
def test_the_ordinals_are_accepted_on_the_row_wrapper_or_on_the_message(browser, mode):
    """aria-posinset may sit on the row wrapper or on the message; the gate must accept either placement."""
    page = _page(browser, mode)
    got, log = _lines()
    try:
        assert page.evaluate("() => window.__sb.dom.threadTotal()") == MESSAGES
        r = wait_for_thread_ready(
            page,
            MESSAGES,
            marker = turn_marker(TURNS - 1, TURNS - 1),
            mode = MODE_WINDOWED,
            timeout_s = 20,
            log = log,
        )
    finally:
        page.close()
    assert r.ready, r.reason
    assert r.probe["setsize"] == MESSAGES
    assert r.conditions["posinset_on_every_row"] is True


def _completeness(browser, mode: str, **kwargs) -> tuple[dict, list[str]]:
    """Bring `mode` up in windowed mode, then run the completeness probe over it."""
    page = _page(browser, mode)
    got, log = _lines()
    try:
        wait_for_thread_ready(
            page,
            MESSAGES,
            marker = turn_marker(TURNS - 1, TURNS - 1),
            mode = MODE_WINDOWED,
            timeout_s = 20,
            log = log,
        )
        out = probe_thread_completeness(
            page,
            first_marker = turn_marker(0, 0),
            expected_messages = MESSAGES,
            timeout_s = kwargs.pop("timeout_s", 15),
            log = log,
            **kwargs,
        )
    finally:
        page.close()
    return out, got


def test_a_virtualised_thread_passes_the_completeness_probe(browser):
    out, _ = _completeness(browser, "windowed")
    assert out["head_reached"] is True, out
    # The default 2,000px step jumps straight past this 2,160px thread, so coverage is unmeasured.
    assert out["ordinal_coverage_complete"] is None, out
    assert out["sweep_continuous"] is False
    assert "never in view" in out["coverage_reason"]
    assert out["ordinal_coverage_state"] == COVERAGE_UNMEASURED, out


def test_a_virtualised_thread_covers_every_ordinal_when_the_sweep_is_continuous(browser):
    """Overlapping steps make the sweep continuous, so coverage is measurable and complete."""
    out, _ = _completeness(browser, "windowed", step_px = ROW_PX * 2)
    assert out["head_reached"] is True, out
    assert out["sweep_continuous"] is True
    assert out["ordinal_coverage_complete"] is True, out
    assert out["ordinal_coverage_state"] == COVERAGE_COMPLETE, out
    assert out["ordinals_seen_count"] == MESSAGES
    assert out["ordinals_missing"] == []


def test_a_thread_that_lost_the_middle_passes_the_head_marker_and_fails_coverage(browser):
    """A store with only its first and last page passes the head marker; ordinal coverage catches it."""
    out, got = _completeness(browser, "windowed_lost_middle")
    assert out["head_reached"] is True, out
    assert out["ordinal_coverage_complete"] is False, out
    assert out["ordinal_coverage_state"] == COVERAGE_INCOMPLETE, out
    assert out["ordinals_missing"] == list(range(4, MESSAGES - 2))
    assert out["ordinals_in_window_holes"] == list(range(4, MESSAGES - 2))
    assert "MIDDLE" in out["coverage_reason"]
    assert any("COMPLETENESS FAILED" in line for line in got)


def test_coverage_does_not_apply_to_an_arm_that_publishes_no_ordinals(browser):
    """A fully mounted arm publishes no aria-posinset; coverage is NOT APPLICABLE there, not UNMEASURED."""
    page = _page(browser, "full")
    got, log = _lines()
    try:
        wait_for_thread_ready(
            page,
            MESSAGES,
            marker = turn_marker(TURNS - 1, TURNS - 1),
            mode = MODE_FULL,
            timeout_s = 20,
            log = log,
        )
        out = probe_thread_completeness(
            page,
            first_marker = turn_marker(0, 0),
            expected_messages = MESSAGES,
            timeout_s = 10,
            log = log,
        )
    finally:
        page.close()
    assert out["head_reached"] is True, out
    assert out["ordinals_seen_count"] == 0
    assert out["ordinal_coverage_complete"] is None, out
    assert out["ordinal_coverage_state"] == COVERAGE_NOT_APPLICABLE, out
    assert "nothing to count" in out["coverage_reason"]


def test_coverage_is_not_measured_when_the_gesture_never_reached_the_top(browser):
    """Coverage is unmeasured, not missing, when the scroll gesture never reached the top of the thread."""
    out, got = _completeness(
        browser,
        "windowed",
        steps = 1,
        step_px = ROW_PX * 2,
        timeout_s = 2,
    )
    assert out["head_reached"] is None, out
    assert out["reached_top"] is False
    assert out["ordinal_coverage_complete"] is None, out
    assert out["ordinal_coverage_state"] == COVERAGE_UNMEASURED, out
    assert "never looked for" in out["coverage_reason"]
    assert any("NOT MEASURED" in line for line in got)


def test_windowed_mode_also_admits_a_thread_short_enough_to_mount_whole(browser):
    """A windowed arm on a thread that fits in the window is admitted only when every message is mounted."""
    page = _page(browser, "full", turns = 2)
    got, log = _lines()
    try:
        r = wait_for_thread_ready(
            page,
            4,
            marker = turn_marker(1, 1),
            mode = MODE_WINDOWED,
            timeout_s = 20,
            log = log,
        )
    finally:
        page.close()
    assert r.ready, r.reason
    assert r.probe["setsize"] is None
    assert r.conditions["total_declared"] is True
    assert r.probe["posinset_count"] == 0
    assert r.conditions["posinset_ordinals_valid"] is True
    assert r.conditions["posinset_reaches_end"] is True


def test_a_half_mounted_thread_is_refused_in_full_mode(browser):
    """The original failure, reproduced: mounting, not finished, and not admitted."""
    page = _page(browser, "mounting")
    got, log = _lines()
    try:
        with pytest.raises(ThreadNotReady) as caught:
            wait_for_thread_ready(
                page,
                MESSAGES,
                marker = turn_marker(TURNS - 1, TURNS - 1),
                mode = MODE_FULL,
                timeout_s = 4,
                log = log,
            )
    finally:
        page.close()
    detail = caught.value.detail
    assert detail["ready"] is False
    assert detail["conditions"]["all_messages_mounted"] is False
    assert detail["conditions"]["end_present"] is False
    assert detail["probe"]["mounted"] < MESSAGES


def test_a_half_mounted_thread_is_refused_in_windowed_mode_too(browser):
    """A half-mounted thread is refused in windowed mode too, or windowed would switch the gate off."""
    page = _page(browser, "mounting")
    got, log = _lines()
    try:
        with pytest.raises(ThreadNotReady) as caught:
            wait_for_thread_ready(
                page,
                MESSAGES,
                marker = turn_marker(TURNS - 1, TURNS - 1),
                mode = MODE_WINDOWED,
                timeout_s = 4,
                log = log,
            )
    finally:
        page.close()
    conditions = caught.value.detail["conditions"]
    assert conditions["settled"] is False
    assert conditions["end_present"] is False
    assert conditions["total_declared"] is False


def test_a_window_that_publishes_no_total_is_refused(browser):
    page = _page(browser, "windowed_no_total")
    got, log = _lines()
    try:
        with pytest.raises(ThreadNotReady) as caught:
            wait_for_thread_ready(
                page,
                MESSAGES,
                marker = turn_marker(TURNS - 1, TURNS - 1),
                mode = MODE_WINDOWED,
                timeout_s = 4,
                log = log,
            )
    finally:
        page.close()
    conditions = caught.value.detail["conditions"]
    assert conditions["total_declared"] is False
    assert conditions["total_matches_seeded"] is False
    assert conditions["settled"] is True
    assert conditions["end_present"] is True


def test_a_window_over_the_wrong_end_of_the_thread_is_refused(browser):
    page = _page(browser, "windowed_at_top")
    got, log = _lines()
    try:
        with pytest.raises(ThreadNotReady) as caught:
            wait_for_thread_ready(
                page,
                MESSAGES,
                marker = turn_marker(TURNS - 1, TURNS - 1),
                mode = MODE_WINDOWED,
                timeout_s = 4,
                log = log,
            )
    finally:
        page.close()
    conditions = caught.value.detail["conditions"]
    assert conditions["end_present"] is False
    assert conditions["anchored_at_end"] is False
    assert conditions["total_matches_seeded"] is True


def _refused(browser, mode: str) -> dict:
    """Run the gate against `mode` in windowed mode and return the conditions it refused on."""
    page = _page(browser, mode)
    got, log = _lines()
    try:
        with pytest.raises(ThreadNotReady) as caught:
            wait_for_thread_ready(
                page,
                MESSAGES,
                marker = turn_marker(TURNS - 1, TURNS - 1),
                mode = MODE_WINDOWED,
                timeout_s = 4,
                log = log,
            )
    finally:
        page.close()
    return caught.value.detail


def test_a_window_whose_rows_all_publish_a_zero_ordinal_is_refused(browser):
    """aria-posinset is 1-based: rows publishing 0 are not positions, so an all-zero window is refused."""
    detail = _refused(browser, "windowed_zero_ordinals")
    conditions = detail["conditions"]
    assert conditions["posinset_on_every_row"] is True
    assert detail["probe"]["posinset_count"] == WINDOW
    assert conditions["posinset_ordinals_valid"] is False
    assert detail["probe"]["min_posinset"] == 0
    assert conditions["settled"] is True
    assert conditions["end_present"] is True
    assert conditions["total_matches_seeded"] is True


def test_a_window_whose_rows_all_claim_the_same_ordinal_is_refused(browser):
    """Ordinals must be unique, because they map each row to its place; a shared ordinal is refused."""
    detail = _refused(browser, "windowed_duplicate_ordinals")
    conditions = detail["conditions"]
    assert conditions["posinset_on_every_row"] is True
    assert detail["probe"]["posinset_count"] == WINDOW
    assert detail["probe"]["posinset_distinct"] == 1
    assert conditions["posinset_ordinals_valid"] is False
    assert conditions["posinset_reaches_end"] is True
    assert conditions["end_present"] is True


def test_a_bottom_window_numbered_from_one_is_refused(browser):
    """A bottom window numbered from 1 is refused: it would claim to be the thread's first messages."""
    detail = _refused(browser, "windowed_from_one")
    conditions = detail["conditions"]
    assert conditions["posinset_on_every_row"] is True
    assert conditions["posinset_ordinals_valid"] is True
    assert detail["probe"]["max_posinset"] == WINDOW
    assert conditions["posinset_reaches_end"] is False
    assert conditions["end_present"] is True
    assert conditions["anchored_at_end"] is True


def test_a_thread_that_lost_its_head_passes_readiness_and_fails_completeness(browser):
    """Readiness cannot tell a lost head from a virtualizer; only the completeness probe catches it."""
    page = _page(browser, "windowed_lost_head")
    got, log = _lines()
    try:
        r = wait_for_thread_ready(
            page,
            MESSAGES,
            marker = turn_marker(TURNS - 1, TURNS - 1),
            mode = MODE_WINDOWED,
            timeout_s = 20,
            log = log,
        )
        assert r.ready
        out = probe_thread_completeness(
            page,
            first_marker = turn_marker(0, 0),
            expected_messages = MESSAGES,
            timeout_s = 6,
            log = log,
        )
    finally:
        page.close()
    assert out["head_reached"] is False, out
    assert "not holding the whole conversation" in out["reason"]
    assert any("COMPLETENESS FAILED" in line for line in got)


def test_evaluate_never_reports_a_mode_inapplicable_condition_as_a_pass():
    """`None` is not `True`, and the difference is the whole design of the two modes."""
    probe = {
        "probe_attempted": True,
        "mounted": 6,
        "elements": 40,
        "composer": True,
        "setsize": None,
        "posinset_count": 0,
        "marker_found": True,
        "marker_from_end": 1,
        "scroll_height": 100,
        "from_bottom": 0,
        "app_says_at_bottom": True,
        "pinning": False,
    }
    full = evaluate(probe, probe, 18, MODE_FULL)
    assert full["total_declared"] is None
    assert full["all_messages_mounted"] is False
    windowed = evaluate(probe, probe, 18, MODE_WINDOWED)
    assert windowed["total_declared"] is False
    assert "all_messages_mounted" not in windowed


def _windowed_probe(**changes) -> dict:
    """A settled window at the end of an 18-message thread, correct in every respect."""
    probe = {
        "probe_attempted": True,
        "mounted": 6,
        "elements": 40,
        "composer": True,
        "setsize": 18,
        "posinset_count": 6,
        "posinset_distinct": 6,
        "min_posinset": 13,
        "max_posinset": 18,
        "marker_found": True,
        "marker_from_end": 1,
        "scroll_height": 100,
        "from_bottom": 0,
        "app_says_at_bottom": True,
        "pinning": False,
    }
    probe.update(changes)
    return probe


def test_evaluate_refuses_ordinals_that_are_not_positions():
    """The three malformed shapes, on the decision function itself.

    The live tests above put each of these in a real browser; this pins the rule they are being
    judged by, including which of the two conditions each one trips.
    """
    good = _windowed_probe()
    assert evaluate(good, good, 18, MODE_WINDOWED)["posinset_ordinals_valid"] is True
    zeros = _windowed_probe(posinset_distinct = 1, min_posinset = 0, max_posinset = 0)
    assert evaluate(zeros, zeros, 18, MODE_WINDOWED)["posinset_ordinals_valid"] is False
    duplicates = _windowed_probe(posinset_distinct = 1, min_posinset = 18)
    assert evaluate(duplicates, duplicates, 18, MODE_WINDOWED)["posinset_ordinals_valid"] is False
    from_one = _windowed_probe(min_posinset = 1, max_posinset = 6)
    conditions = evaluate(from_one, from_one, 18, MODE_WINDOWED)
    assert conditions["posinset_ordinals_valid"] is True
    assert conditions["posinset_reaches_end"] is False
    over = _windowed_probe(max_posinset = 19)
    assert evaluate(over, over, 18, MODE_WINDOWED)["posinset_ordinals_valid"] is False
    assert evaluate(zeros, zeros, 18, MODE_FULL)["posinset_ordinals_valid"] is None


def test_evaluate_does_not_waive_malformed_ordinals_for_a_fully_mounted_thread():
    """Mounting every row does not waive malformed ordinals; the waiver covers only a thread with none."""
    silent = _windowed_probe(
        mounted = 18,
        setsize = None,
        posinset_count = 0,
        posinset_distinct = 0,
        min_posinset = None,
        max_posinset = None,
    )
    conditions = evaluate(silent, silent, 18, MODE_WINDOWED)
    assert conditions["posinset_ordinals_valid"] is True
    assert conditions["posinset_reaches_end"] is True
    junk = _windowed_probe(
        mounted = 18,
        posinset_count = 18,
        posinset_distinct = 1,
        min_posinset = 0,
        max_posinset = 0,
    )
    conditions = evaluate(junk, junk, 18, MODE_WINDOWED)
    assert conditions["posinset_ordinals_valid"] is False
    assert conditions["posinset_reaches_end"] is False


def test_ordinal_coverage_never_reports_a_gap_in_the_gesture_as_data_loss():
    """Missing ordinals are NOT MEASURED, not MISSING, when the traversal's stops did not overlap."""
    coarse = {
        "reached_target": True,
        "ordinals_seen": [1, 2, 3, 16, 17, 18],
        "ordinals_in_window_holes": [],
        "sweep_continuous": False,
        "traversal_stops": 2,
    }
    got_coarse = ordinal_coverage(coarse, 18)
    assert got_coarse["ordinal_coverage_complete"] is None
    assert got_coarse["ordinal_coverage_state"] == COVERAGE_UNMEASURED
    continuous = dict(coarse, sweep_continuous = True)
    got = ordinal_coverage(continuous, 18)
    assert got["ordinal_coverage_complete"] is False
    assert got["ordinal_coverage_state"] == COVERAGE_INCOMPLETE
    assert got["ordinals_missing"] == list(range(4, 16))
    assert got["ordinals_missing_count"] == 12


def test_ordinal_coverage_reports_a_hole_inside_one_mounted_window_whatever_the_step():
    """A hole inside one mounted window is reported at any step size, unless another stop saw the row."""
    lost_middle = {
        "reached_target": True,
        "ordinals_seen": [1, 2, 3, 16, 17, 18],
        "ordinals_in_window_holes": list(range(4, 16)),
        "sweep_continuous": False,
        "traversal_stops": 2,
    }
    got = ordinal_coverage(lost_middle, 18)
    assert got["ordinal_coverage_complete"] is False
    assert "MIDDLE" in got["coverage_reason"]
    late = {
        "reached_target": True,
        "ordinals_seen": list(range(1, 19)),
        "ordinals_in_window_holes": [7, 8],
        "sweep_continuous": True,
        "traversal_stops": 9,
    }
    assert ordinal_coverage(late, 18)["ordinal_coverage_complete"] is True


def test_ordinal_coverage_is_unmeasured_when_the_traversal_never_reached_the_top():
    """Even a completely covered union proves nothing if the gesture stopped short: the rows it
    did not reach are rows it did not look at."""
    stopped = {
        "reached_target": False,
        "ordinals_seen": list(range(1, 19)),
        "ordinals_in_window_holes": [],
        "sweep_continuous": True,
        "traversal_stops": 3,
    }
    got = ordinal_coverage(stopped, 18)
    assert got["ordinal_coverage_complete"] is None
    assert got["ordinal_coverage_state"] == COVERAGE_UNMEASURED
    assert "never looked for" in got["coverage_reason"]


def test_ordinal_coverage_separates_a_question_that_does_not_apply_from_one_it_could_not_answer():
    """Unmeasured ordinals are refused by thread_complete, while an arm that publishes none passes it."""
    walked = {
        "reached_target": True,
        "ordinals_in_window_holes": [],
        "sweep_continuous": False,
        "traversal_stops": 2,
    }
    applies = ordinal_coverage(dict(walked, ordinals_seen = [1, 2, 3, 16, 17, 18]), 18)
    does_not = ordinal_coverage(dict(walked, ordinals_seen = []), 18)
    assert applies["ordinal_coverage_complete"] is does_not["ordinal_coverage_complete"] is None
    assert applies["ordinal_coverage_state"] == COVERAGE_UNMEASURED
    assert does_not["ordinal_coverage_state"] == COVERAGE_NOT_APPLICABLE


def test_evaluate_cannot_settle_on_a_single_sample():
    """One reading is a snapshot, and a snapshot of a growing thread looks exactly like a settled
    one. The first sample can never report settled, whatever it contains."""
    probe = {"probe_attempted": True, "mounted": 18, "elements": 100, "scroll_height": 10}
    assert evaluate(probe, None, 18, MODE_FULL)["settled"] is False
    assert evaluate(probe, probe, 18, MODE_FULL)["settled"] is True
    grew = dict(probe, elements = 101)
    assert evaluate(grew, probe, 18, MODE_FULL)["settled"] is False


def test_a_windowed_thread_with_no_viewport_is_refused(browser):
    """Regression: no viewport must refuse the cell; .aui-thread-scroll-to-bottom is document-scoped."""
    page = _page(browser, "windowed")
    got, log = _lines()
    try:
        renamed = page.evaluate(
            """() => {
                 const vp = document.querySelector(".aui-thread-viewport");
                 if (!vp) return false;
                 vp.classList.remove("aui-thread-viewport");
                 vp.classList.add("aui-thread-scroller");
                 return true;
               }"""
        )
        assert renamed, "the fixture has no viewport to rename"
        with pytest.raises(ThreadNotReady) as caught:
            wait_for_thread_ready(
                page,
                MESSAGES,
                marker = turn_marker(TURNS - 1, TURNS - 1),
                mode = MODE_WINDOWED,
                timeout_s = 3,
                log = log,
            )
    finally:
        page.close()

    assert "viewport_present" in str(caught.value), str(caught.value)
