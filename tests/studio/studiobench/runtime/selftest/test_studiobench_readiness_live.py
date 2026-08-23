# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE GATE, SHOWN PASSING AND SHOWN FAILING, in a real browser.

A gate nobody has watched refuse anything is not a gate, and the one this replaces was refusing
the wrong thing: it counted mounted `[data-role]` nodes, so a thread that mounts a window on
purpose could never satisfy it and the virtualization arm scored UNSCORED. The fix is only worth
having if it can be shown to admit the arm AND still refuse a thread that is not ready, so both
are constructed here rather than argued about.

Ten threads, one gate:

  full                admitted in `full` mode. The shipped app.
  windowed            admitted in `windowed` mode. A window at the end of the thread that
                      publishes aria-setsize and aria-posinset, sits at the bottom, and
                      materialises the head when you scroll to the top.
  mounting            REFUSED in both modes. Nine of eighteen and climbing: the exact state the
                      gate exists for, and the state the old gate did correctly catch.
  windowed_no_total   REFUSED in `windowed` mode. A window that never says how long the thread is.
  windowed_at_top     REFUSED in `windowed` mode. Settled, correct total, and showing the wrong
                      end of the conversation.
  windowed_zero_ordinals
  windowed_duplicate_ordinals
  windowed_from_one   REFUSED in `windowed` mode. All three publish aria-posinset on every mounted
                      row, which is all the gate used to ask for, and none of the three publishes
                      a POSITION: all zeros, all identical, and a window at the bottom of an
                      eighteen-message thread numbered 1..6.
  windowed_lost_head  ADMITTED by the readiness gate and REFUSED by the completeness probe. The
                      honest split, and it is asserted in both directions: standing at the bottom
                      of a thread there is no way to tell a virtualizer from a thread that has
                      lost its history, so the probe walks to the top and looks.
  windowed_lost_middle
                      ADMITTED by the readiness gate, ADMITTED by the head marker, and REFUSED on
                      ordinal coverage. The head is there and the tail is there, so every check
                      that looks at one end of the thread is satisfied; what is gone is the
                      middle, and the only evidence is the ordinals of the mounted rows.

And the coverage verdict is asserted in its NOT MEASURED direction too, twice, because a probe
that reports what it did not look at as data loss is worse than no probe: an arm that publishes no
ordinals at all, and a gesture that never reached the top.

Everything under test is the production code path: the real `scene/dom.js`, the real `PROBE_JS`,
the real `wait_for_thread_ready` and `probe_thread_completeness`. The only synthetic part is the
page, which supplies the DOM contract the app publishes and nothing else.

Requires Playwright with Chromium. Skips cleanly without it, because a machine that cannot run a
browser should say so rather than fail.

    python -m pytest tests/studio/studiobench/runtime/selftest/test_studiobench_readiness_live.py -q
"""

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
MESSAGES = TURNS * 2  # 18, the number in the failure this work exists to fix
WINDOW = 6
#: The fixture's row height. The completeness tests step the traversal by two rows, because a
#: gesture whose stops do not overlap can only report NOT MEASURED, and a test that wants to see
#: coverage refuse something has to give the probe a sweep that actually covers the thread.
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


# ── the fixture itself, before it is trusted to prove anything ──────


def test_the_fixture_marker_matches_the_seeder_exactly(browser):
    """If these two ever drift, every gate below passes or fails for the wrong reason.

    The gate looks for a string the seeder wrote. A fixture that writes a slightly different one
    would make the negative cases pass for free and the positive ones fail mysteriously, so the
    agreement is asserted rather than assumed.
    """
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
        # No aria-setsize anywhere: it must degrade to exactly today's messageCount().
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


# ── what the gate must ADMIT ────────────────────────────────────────


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
    # AND THE ORDINAL CONDITIONS ARE NOT APPLICABLE HERE, which is not the same as passing.
    # Studio publishes no aria-posinset anywhere, so a `full` arm has none to validate and must
    # never be gated on them; `None` is the value the parity layer and this gate both use for a
    # surface that was not measured rather than one that agreed.
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
    # The ordinals of a window at the END of an eighteen-message thread: 13..18, distinct, one per
    # mounted row. This is the shape the three malformed modes below fail to produce.
    assert (r.probe["min_posinset"], r.probe["max_posinset"]) == (MESSAGES - WINDOW + 1, MESSAGES)
    assert r.probe["posinset_distinct"] == WINDOW


@pytest.mark.parametrize("mode", ["windowed", "windowed_flat"])
def test_the_ordinals_are_accepted_on_the_row_wrapper_or_on_the_message(browser, mode):
    """WHERE the attributes live must not decide whether the arm can be scored.

    `thread-message-virtualizer.tsx` renders an absolutely positioned wrapper per item and mounts
    the message inside it, so the element that is a member of the set is the wrapper. That is the
    correct place for `aria-posinset`, and a gate that only looked at `[data-role]` would refuse a
    correctly implemented arm for putting the attribute exactly where it belongs.
    """
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
    # AND THE COVERAGE VERDICT IS NOT MEASURED AT THE DEFAULT STEP, which is the honest answer
    # rather than a flattering one. The gesture jumps 2,000px at a time and this fixture's whole
    # thread is 2,160px, so it lands at the bottom and then at the top and the rows between the
    # two stops were never in view. Nothing is known about them, so nothing is claimed.
    assert out["ordinal_coverage_complete"] is None, out
    assert out["sweep_continuous"] is False
    assert "never in view" in out["coverage_reason"]
    # WHICH KIND OF NOT MEASURED. The arm publishes ordinals, so the question applies and the
    # sweep failed to answer it, which `record_completeness_gate` refuses to score the cell on.
    # The remedy is the smaller step the next test uses, not a gate that accepts the unknown.
    assert out["ordinal_coverage_state"] == COVERAGE_UNMEASURED, out


def test_a_virtualised_thread_covers_every_ordinal_when_the_sweep_is_continuous(browser):
    """The same correct thread, walked in steps small enough to overlap.

    This is what coverage looks like when it is actually measurable: every consecutive stop mounts
    a window overlapping the last, so the union is everything the thread can show, and it is all
    eighteen messages.
    """
    out, _ = _completeness(browser, "windowed", step_px = ROW_PX * 2)
    assert out["head_reached"] is True, out
    assert out["sweep_continuous"] is True
    assert out["ordinal_coverage_complete"] is True, out
    assert out["ordinal_coverage_state"] == COVERAGE_COMPLETE, out
    assert out["ordinals_seen_count"] == MESSAGES
    assert out["ordinals_missing"] == []


def test_a_thread_that_lost_the_middle_passes_the_head_marker_and_fails_coverage(browser):
    """THE CASE THE MARKER CHECK ALONE CALLED COMPLETE.

    The store kept the first page and the last page. Standing at the bottom, every readiness
    condition holds -- the window is at the end, the total is right, the ordinals are positions.
    Scroll to the top and the first message of the conversation arrives, so `head_reached` is
    true. Twelve of the eighteen messages do not exist anywhere in the arm, and before this the
    cell was scoreable.

    What catches it does not depend on the step size: a virtualizer mounts a CONTIGUOUS run, so
    ordinals 4..15 missing from a single mounted window that spans 1..18 is the store, not the
    gesture. That is why this runs at the default step and still refuses.
    """
    out, got = _completeness(browser, "windowed_lost_middle")
    assert out["head_reached"] is True, out
    assert out["ordinal_coverage_complete"] is False, out
    assert out["ordinal_coverage_state"] == COVERAGE_INCOMPLETE, out
    assert out["ordinals_missing"] == list(range(4, MESSAGES - 2))
    assert out["ordinals_in_window_holes"] == list(range(4, MESSAGES - 2))
    assert "MIDDLE" in out["coverage_reason"]
    assert any("COMPLETENESS FAILED" in line for line in got)


def test_coverage_does_not_apply_to_an_arm_that_publishes_no_ordinals(browser):
    """A fully mounted arm publishes no aria-posinset, and that is not eighteen lost messages.

    The probe can be pointed at a `full` arm (`--completeness-probe` is a per-runner flag, not a
    property of the mode), and there is nothing to count when it is. Reporting the seeded ordinals
    as missing there would turn the shipped build into the worst data-loss finding in the payload.

    NOT APPLICABLE rather than UNMEASURED, and the distinction is load-bearing: the gate declines
    to score a cell whose coverage was unmeasured, so calling this one unmeasured would fail the
    shipped build on every cell it is pointed at.
    """
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
    """The rule `head_reached` already follows, applied to coverage.

    One step of two rows on a thread eighteen rows long: the viewport never gets near the top, so
    the head did not mount and most ordinals were never seen. Neither of those is a fact about the
    arm, and both used to be reportable as one.
    """
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
    """The same arm at a rung whose thread fits inside the window.

    A virtualised build mounts every message of a short thread, exactly like the shipped one, and
    then has nothing to publish a total ABOUT. If `windowed` refused that, the mode would only
    work at some rungs of the same arm and the ladder would have holes in it for a reason that has
    nothing to do with the app. The relaxation needs `mounted >= expected`, which IS the
    full-mount condition, so nothing half-built can reach it -- the test below proves that
    directly.
    """
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
    # THE SAME WAIVER COVERS THE ORDINALS, and for the same reason: there are none to publish and
    # none to validate. It is waived on `mounted >= expected` AND on the absence of ordinals
    # together, so an arm that publishes malformed ones cannot buy its way out by also mounting
    # the whole thread.
    assert r.probe["posinset_count"] == 0
    assert r.conditions["posinset_ordinals_valid"] is True
    assert r.conditions["posinset_reaches_end"] is True


# ── what the gate must REFUSE ───────────────────────────────────────


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
    # And the two NEW conditions caught it too, independently of the count. That matters: it is
    # what makes the windowed mode safe, because the windowed mode has no count to rely on.
    assert detail["conditions"]["end_present"] is False
    assert detail["probe"]["mounted"] < MESSAGES


def test_a_half_mounted_thread_is_refused_in_windowed_mode_too(browser):
    """The one that decides whether `windowed` is a gate or a hole.

    A thread that is still mounting looks superficially like a windowed one: fewer messages are in
    the DOM than the thread contains. If `windowed` mode admitted it, the mode would be a way of
    switching the gate off, and every reading taken through it would be the flattering garbage the
    gate exists to prevent.
    """
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
    # THREE independent refusals, not one. It is still growing, it has not reached the end of the
    # thread, and it publishes no total.
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
    # Everything else about it was fine, which is the point: the refusal is specific.
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
    """`aria-posinset` is 1-based, so 0 is not a position, it is the attribute being present.

    This is the first of the three shapes that passed the old condition: it asked whether every
    mounted row carried a finite number and every row here does. A window numbered 0,0,0,0,0,0
    tells a screen reader nothing about where it is in the thread, and told this gate nothing
    either while satisfying it.
    """
    detail = _refused(browser, "windowed_zero_ordinals")
    conditions = detail["conditions"]
    # The OLD condition still passes, which is exactly why it was not a check.
    assert conditions["posinset_on_every_row"] is True
    assert detail["probe"]["posinset_count"] == WINDOW
    assert conditions["posinset_ordinals_valid"] is False
    assert detail["probe"]["min_posinset"] == 0
    # And nothing else about the thread was wrong: the refusal is specific to the ordinals.
    assert conditions["settled"] is True
    assert conditions["end_present"] is True
    assert conditions["total_matches_seeded"] is True


def test_a_window_whose_rows_all_claim_the_same_ordinal_is_refused(browser):
    """Six mounted rows and one position between them.

    Uniqueness is the property that makes the ordinals a MAP from row to place in the thread. Note
    what this mode gets right, so the refusal cannot be credited to anything else: the ordinal it
    publishes is the seeded total, so the window still reaches the end of the thread and
    `posinset_reaches_end` passes.
    """
    detail = _refused(browser, "windowed_duplicate_ordinals")
    conditions = detail["conditions"]
    assert conditions["posinset_on_every_row"] is True
    assert detail["probe"]["posinset_count"] == WINDOW
    assert detail["probe"]["posinset_distinct"] == 1
    assert conditions["posinset_ordinals_valid"] is False
    assert conditions["posinset_reaches_end"] is True
    assert conditions["end_present"] is True


def test_a_bottom_window_numbered_from_one_is_refused(browser):
    """The likeliest of the three to be written by accident: the index WITHIN the window.

    Every ordinal here is a legal position -- 1..6, distinct, inside the declared set size -- so
    validity alone admits it. What refuses it is that a window sitting at the bottom of an
    eighteen-message thread claims to be its first six messages, which would make the mounted set
    unlocatable and would make a window at the end indistinguishable from one at the start.
    """
    detail = _refused(browser, "windowed_from_one")
    conditions = detail["conditions"]
    assert conditions["posinset_on_every_row"] is True
    assert conditions["posinset_ordinals_valid"] is True
    assert detail["probe"]["max_posinset"] == WINDOW
    assert conditions["posinset_reaches_end"] is False
    # The TEXT of the last message is mounted; only the numbering disagrees with it. That split is
    # the point: `end_present` reads the thread and this reads what the arm says about it.
    assert conditions["end_present"] is True
    assert conditions["anchored_at_end"] is True


def test_a_thread_that_lost_its_head_passes_readiness_and_fails_completeness(browser):
    """THE HONEST SPLIT, asserted in both directions.

    From the bottom of the thread this is indistinguishable from a correct virtualizer: the same
    window, the same published total, the same anchor. The readiness gate therefore admits it, and
    saying otherwise would be claiming a power the reading does not have. What catches it is the
    completeness probe, which does the only thing a user could do -- scroll to the top and look for
    the beginning of the conversation.
    """
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


# ── the decision function, without a browser ────────────────────────


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
    # PAST THE END OF THE SET THE SAME ROWS DECLARE. 19 of 18 is not a position either, and it is
    # what an off-by-one in a virtualizer's index-to-ordinal arithmetic produces.
    over = _windowed_probe(max_posinset = 19)
    assert evaluate(over, over, 18, MODE_WINDOWED)["posinset_ordinals_valid"] is False
    # And every one of them is NOT APPLICABLE in full mode, where nothing publishes ordinals.
    assert evaluate(zeros, zeros, 18, MODE_FULL)["posinset_ordinals_valid"] is None


def test_evaluate_does_not_waive_malformed_ordinals_for_a_fully_mounted_thread():
    """The waiver is for a thread that publishes NO ordinals, not for one that publishes junk.

    A short thread mounted whole publishes nothing and is admitted, which is what keeps a windowed
    arm scoreable at the small rungs. An arm that publishes an ordinal of 0 on every row is broken
    for a screen reader whether or not the thread happened to fit in the window, and mounting
    everything must not be a way to skip the check.
    """
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


# ── the coverage verdict, without a browser ─────────────────────────


def test_ordinal_coverage_never_reports_a_gap_in_the_gesture_as_data_loss():
    """NOT MEASURED and MISSING are different answers, and the difference is the whole probe.

    Same missing ordinals, same reached top, and the only thing that changes is whether the stops
    of the traversal overlapped each other. When they did not, the rows between two stops were
    never mounted by anybody and the probe has nothing to say about them.
    """
    coarse = {
        "reached_target": True,
        "ordinals_seen": [1, 2, 3, 16, 17, 18],
        "ordinals_in_window_holes": [],
        "sweep_continuous": False,
        "traversal_stops": 2,
    }
    got_coarse = ordinal_coverage(coarse, 18)
    assert got_coarse["ordinal_coverage_complete"] is None
    # And it is the kind of None that is NOT a pass: the ordinals apply to this arm and the sweep
    # did not inspect them. `not_applicable` -- the other None -- is an arm publishing no ordinals
    # at all, which is the case below in `test_ordinal_coverage_separates_...`.
    assert got_coarse["ordinal_coverage_state"] == COVERAGE_UNMEASURED
    continuous = dict(coarse, sweep_continuous = True)
    got = ordinal_coverage(continuous, 18)
    assert got["ordinal_coverage_complete"] is False
    assert got["ordinal_coverage_state"] == COVERAGE_INCOMPLETE
    assert got["ordinals_missing"] == list(range(4, 16))
    assert got["ordinals_missing_count"] == 12


def test_ordinal_coverage_reports_a_hole_inside_one_mounted_window_whatever_the_step():
    """The reading that does not depend on how coarsely the thread was walked.

    A virtualizer mounts a contiguous run, so an ordinal missing from between the smallest and the
    largest mounted at a single stop was not skipped by the gesture. It is intersected with "never
    seen anywhere", so a row that was still materialising at one stop and mounted at the next is
    not reported as lost.
    """
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
    """THE TWO KINDS OF `None`, side by side, on traversals that differ in one thing.

    Both sweeps reached the top and both are too coarse to overlap. One is walking an arm that
    publishes ordinals and cannot say what happened to twelve of them; the other is walking an arm
    that publishes none at all, which is the shipped build, where there is nothing to say. The
    `thread_complete` gate refuses to score the first and passes the second, so a single `None`
    covering both is a gate that has to choose which mistake to make.
    """
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


# ── the viewport itself, asserted rather than inferred ────────────────────────


def test_a_windowed_thread_with_no_viewport_is_refused(browser):
    """REGRESSION. Every other windowed condition degrades to a pass when the scroller is gone.

    `from_bottom` is null so the arithmetic is skipped, and the app's own answer is read off
    `.aui-thread-scroll-to-bottom`, which is a DESCENDANT of the viewport but is looked up at
    DOCUMENT scope. Renaming the viewport class therefore leaves that control reachable,
    `app_says_at_bottom` true and `anchored_at_end` true, and the cell was admitted with no
    viewport at all: the completeness probe then returns `probe_attempted: false`, the scroll
    actions return `not_run`, a not-run action blanks only its own timings, and the census viewport
    fields go null. Nothing refused, so the film was scored without the surface it was measuring.
    """
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

    # Named in the refusal, so the reader is told the surface is missing rather than left to infer
    # it from a condition that is really about something else.
    assert "viewport_present" in str(caught.value), str(caught.value)
