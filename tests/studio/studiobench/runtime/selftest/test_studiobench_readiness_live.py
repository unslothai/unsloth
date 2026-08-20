# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE GATE, SHOWN PASSING AND SHOWN FAILING, in a real browser.

A gate nobody has watched refuse anything is not a gate, and the one this replaces was refusing
the wrong thing: it counted mounted `[data-role]` nodes, so a thread that mounts a window on
purpose could never satisfy it and the virtualization arm scored UNSCORED. The fix is only worth
having if it can be shown to admit the arm AND still refuse a thread that is not ready, so both
are constructed here rather than argued about.

Six threads, one gate:

  full                admitted in `full` mode. The shipped app.
  windowed            admitted in `windowed` mode. A window at the end of the thread that
                      publishes aria-setsize and aria-posinset, sits at the bottom, and
                      materialises the head when you scroll to the top.
  mounting            REFUSED in both modes. Nine of eighteen and climbing: the exact state the
                      gate exists for, and the state the old gate did correctly catch.
  windowed_no_total   REFUSED in `windowed` mode. A window that never says how long the thread is.
  windowed_at_top     REFUSED in `windowed` mode. Settled, correct total, and showing the wrong
                      end of the conversation.
  windowed_lost_head  ADMITTED by the readiness gate and REFUSED by the completeness probe. The
                      honest split, and it is asserted in both directions: standing at the bottom
                      of a thread there is no way to tell a virtualizer from a thread that has
                      lost its history, so the probe walks to the top and looks.

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
    MODE_FULL,
    MODE_WINDOWED,
    ThreadNotReady,
    evaluate,
    probe_thread_completeness,
    wait_for_thread_ready,
)
from studiobench.runtime.seeder import turn_marker  # noqa: E402

TURNS = 9
MESSAGES = TURNS * 2  # 18, the number in the failure this work exists to fix
WINDOW = 6

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
    assert r.conditions["anchored_at_end"] is True
    assert r.conditions["end_present"] is True


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


def test_a_virtualised_thread_passes_the_completeness_probe(browser):
    page = _page(browser, "windowed")
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
            timeout_s = 15,
            log = log,
        )
    finally:
        page.close()
    assert out["head_reached"] is True, out


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


def test_evaluate_cannot_settle_on_a_single_sample():
    """One reading is a snapshot, and a snapshot of a growing thread looks exactly like a settled
    one. The first sample can never report settled, whatever it contains."""
    probe = {"probe_attempted": True, "mounted": 18, "elements": 100, "scroll_height": 10}
    assert evaluate(probe, None, 18, MODE_FULL)["settled"] is False
    assert evaluate(probe, probe, 18, MODE_FULL)["settled"] is True
    grew = dict(probe, elements = 101)
    assert evaluate(grew, probe, 18, MODE_FULL)["settled"] is False
