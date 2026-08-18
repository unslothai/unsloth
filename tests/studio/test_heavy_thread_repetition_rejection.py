# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A repetition whose action never happened must not be aggregated away.

`test_heavy_thread_harness_contract.py` pins the SHAPE of the harness: every recorded metric is
printed, no verdict rests on a Chromium-only counter. This file pins the ARITHMETIC of the one
place three repetitions become one number, which is the other way a measurement harness goes
false-green -- it drives the page, it prints a plausible table, and one of the three columns
behind each cell is of an interaction that never occurred.

The defect this file exists to keep out, previously live: `median()` filtering out the `None` a
timed-out repetition reports, so "the menu never opened once in three tries" is published as a
clean median of three, and the verdict's `openMs is None` check then reads the median of the
repetitions that did work.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_harness():
    """Import the harness module without needing a browser.

    The module imports `playwright.sync_api` at the top for `sync_playwright`, which nothing in
    this file calls. Stubbing it keeps these tests runnable in the CPU test job, where the
    Playwright package is not installed, rather than skipping the arithmetic along with the
    browser.
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
    import playwright_heavy_thread

    return playwright_heavy_thread


HARNESS = _load_harness()


# ── the aggregation ───────────────────────────────────────────────────


def menu_row(open_ms: float | None) -> dict:
    """One repetition of the menu action. `open_ms is None` is the menu that never opened."""
    close_ms = 40.0
    return {
        "name": "menu",
        "ran": True,
        "openMs": open_ms,
        "closeMs": close_ms,
        "open_close_ms": None if open_ms is None else round(open_ms + close_ms, 1),
        "bodyPointerEvents": "none",
        "bodyPointerEventsAfterClose": "auto",
        "itemsWhileOpen": 5,
        "triggersWhileHovered": 3,
    }


def summarise_menu(rows: list[dict]) -> dict:
    return HARNESS.summarise({"menu": rows})["menu"]


def test_one_repetition_that_never_opened_the_menu_poisons_the_median() -> None:
    # Filtering the null out and taking the median of the rest silently changes the sample
    # population, and the verdict's `openMs is None` check then reads the median of the
    # repetitions that worked.
    merged = summarise_menu([menu_row(80.0), menu_row(None), menu_row(120.0)])
    assert merged["openMs"] is None, merged


def test_a_timing_that_was_null_in_every_repetition_is_still_reported() -> None:
    # Present-and-None, not absent: a key that is missing entirely makes the verdict raise
    # KeyError instead of naming the action that never happened.
    merged = summarise_menu([menu_row(None), menu_row(None), menu_row(None)])
    assert "openMs" in merged and merged["openMs"] is None, merged


def test_the_median_of_three_good_repetitions_is_unchanged() -> None:
    # The guard above must not cost the normal case its median. Expected to pass both before and
    # after the guard: it is a no-regression check and proves nothing about the guard by itself.
    merged = summarise_menu([menu_row(80.0), menu_row(100.0), menu_row(120.0)])
    assert merged["openMs"] == 100.0, merged
    assert merged["closeMs"] == 40.0, merged
    assert merged["open_close_ms"] == 140.0, merged
    assert merged["repetitions"] == 3, merged
    assert merged["per_repetition"] == [120.0, 140.0, 160.0], merged


def test_an_even_number_of_repetitions_still_averages_the_middle_two() -> None:
    # Also expected green in both directions: the guard must not change how a median is taken,
    # only which sets of values are allowed to have one.
    merged = summarise_menu([menu_row(80.0), menu_row(100.0)])
    assert merged["openMs"] == 90.0, merged


# ── the verdict ───────────────────────────────────────────────────────


def clean_action(**extra) -> dict:
    return {
        "ran": True,
        "wall_ms": 100.0,
        "longest_stall_ms": 10.0,
        "worst_frame_ms": 20.0,
        "frames_over_33": 3.0,
        **extra,
    }


VIEWPORT = {"scrollHeight": 20000, "clientHeight": 900, "scrollTop": 0}
COUNTS = {"actionBars": 10}


def clean_actions() -> dict:
    """Six actions that `action_failures()` has nothing to say about."""
    return {
        "keystroke": clean_action(median_sample_ms = 40.0, domText = "aaa", runtimeText = "aaa"),
        "scroll": clean_action(
            gestureMs = 300.0,
            settleMs = 400.0,
            scrolledPx = HARNESS.SCROLL_STEPS * HARNESS.SCROLL_STEP_PX,
            pointer_on_message = True,
            pointer_under = "p",
            pointer_at = "720,450",
        ),
        "jump": clean_action(paintedMs = 90.0, settleMs = 200.0, landedAt = 0, travelledPx = 19000),
        "menu": clean_action(
            openMs = 100.0,
            closeMs = 40.0,
            open_close_ms = 140.0,
            itemsWhileOpen = 5,
            triggersWhileHovered = 3,
            bodyPointerEvents = "none",
            bodyPointerEventsAfterClose = "auto",
        ),
        "delete": clean_action(ms = 120.0, before = 20, after = 19),
        "reopen": clean_action(ms = 500.0, closedMs = 30.0, settleMs = 900.0),
    }


def failures(actions: dict) -> list[str]:
    return HARNESS.action_failures("chromium at 25000 chars (isolated)", actions, COUNTS, VIEWPORT)


def test_a_clean_table_is_not_a_harness_failure() -> None:
    # Expected green in both directions. It is here so that the tests below are known to be
    # reporting the thing they name rather than a fixture the verdict rejects for another reason.
    assert failures(clean_actions()) == []


def test_the_verdict_rejects_a_menu_that_never_opened_in_one_repetition() -> None:
    # The end-to-end shape of the defect: three repetitions, one of which timed out, aggregated
    # and then judged. Before the guard the median of the two that worked reaches the verdict and
    # it has nothing to say.
    actions = clean_actions()
    actions["menu"] = {**summarise_menu([menu_row(80.0), menu_row(None), menu_row(120.0)])}
    assert any("never opened the message action menu" in f for f in failures(actions)), failures(
        actions
    )


def test_the_verdict_rejects_a_delete_that_timed_out_in_one_repetition() -> None:
    rows = [
        {"name": "delete", "ran": True, "ms": ms, "before": 20, "after": 19}
        for ms in (100.0, None, 140.0)
    ]
    actions = clean_actions()
    actions["delete"] = HARNESS.summarise({"delete": rows})["delete"]
    assert any("never deleted a message" in f for f in failures(actions)), failures(actions)


def test_the_verdict_rejects_a_reopen_whose_unmount_was_never_seen() -> None:
    rows = [
        {"name": "reopen", "ran": True, "ms": 500.0, "closedMs": closed, "settleMs": 900.0}
        for closed in (30.0, None, 35.0)
    ]
    actions = clean_actions()
    actions["reopen"] = HARNESS.summarise({"reopen": rows})["reopen"]
    assert any("never saw the thread unmount" in f for f in failures(actions)), failures(actions)


def scroll_row(on_message: bool, under: str) -> dict:
    return {
        "name": "scroll",
        "ran": True,
        "gestureMs": 300.0,
        "settleMs": 400.0,
        "scrolledPx": HARNESS.SCROLL_STEPS * HARNESS.SCROLL_STEP_PX,
        "pointer_on_message": on_message,
        "pointer_under": under,
        "pointer_at": "720,450" if on_message else None,
    }


def test_the_verdict_rejects_a_scroll_whose_pointer_was_off_content_in_one_repetition() -> None:
    # Playwright's mouse starts at (0, 0) on every fresh page, which in this fixture is the
    # scroller's own gutter -- the arm scroll_predecessor_probe.py registers as `gutter_only` and
    # keeps as the artificial control. Measured on this tree at 300K, medians of 3: gutter 7.4ms
    # longest stall and 17.9ms worst frame against 37.3ms and 29.3ms with the pointer on the
    # conversation. So a repetition that scrolled the gutter under-reports two of the four
    # portable primaries by 3-4x and 1.5x, and it is inside the published median.
    #
    # Per repetition, because `summarise` keeps only the last one's copy of a non-numeric proof.
    rows = [
        scroll_row(True, "p"),
        scroll_row(False, "viewport"),
        scroll_row(True, "p"),
    ]
    actions = clean_actions()
    actions["scroll"] = HARNESS.summarise({"scroll": rows})["scroll"]
    assert any(
        "with the pointer off message content on repetition(s) [2]" in f for f in failures(actions)
    ), failures(actions)


def test_the_verdict_rejects_an_action_that_never_settled() -> None:
    # A null settle time is the settle loop giving up, not "this engine does not report that", but
    # it prints as the same `-` and the axis it feeds merely becomes "not recorded" -- so another
    # axis can carry the discrimination check and the run exits 0 having timed out.
    for name in ("scroll", "jump", "reopen"):
        actions = clean_actions()
        actions[name] = {**actions[name], "settleMs": None}
        assert any("never reached a settled state" in f for f in failures(actions)), (
            name,
            failures(actions),
        )


# ── the NUMERIC proofs ────────────────────────────────────────────────
#
# The half of the defect above that median() cannot reach. median() returns None the moment one
# repetition is None, which covers every TIMING, because a timed-out action reports null. It does
# nothing for a proof that is a NUMBER in every repetition and merely the wrong number in one of
# them: the jump that never left the bottom reports `landedAt = bottom`, and the median of
# [0, bottom, 0] is 0, which is exactly what an arrived jump looks like.


def jump_row(landed: float, travelled: float = 19000.0) -> dict:
    return {
        "name": "jump",
        "ran": True,
        "paintedMs": 90.0,
        "settleMs": 200.0,
        "travelledPx": travelled,
        "landedAt": landed,
    }


def test_the_verdict_rejects_a_jump_that_did_not_move_in_one_repetition() -> None:
    # The airtight case: two good repetitions either side of one that never left the bottom.
    rows = [jump_row(0), jump_row(19000.0), jump_row(0)]
    actions = clean_actions()
    actions["jump"] = HARNESS.summarise({"jump": rows})["jump"]
    assert actions["jump"]["landedAt"] == 0, "the median is still the arrived-looking 0"
    assert any("landed at 19000.0px on repetition 2" in f for f in failures(actions)), failures(
        actions
    )


def test_the_verdict_rejects_a_jump_with_nothing_to_jump_through_in_one_repetition() -> None:
    # travelledPx is medianed the same way, so one repetition taken on a collapsed viewport is
    # invisible between two full-height ones.
    rows = [jump_row(0), jump_row(0, travelled = 100.0), jump_row(0)]
    actions = clean_actions()
    actions["jump"] = HARNESS.summarise({"jump": rows})["jump"]
    assert any(
        "had only 100.0px to jump through on repetition 2" in f for f in failures(actions)
    ), failures(actions)


def test_the_verdict_rejects_a_scroll_that_travelled_nothing_in_one_repetition() -> None:
    full = HARNESS.SCROLL_STEPS * HARNESS.SCROLL_STEP_PX
    rows = [
        {**scroll_row(True, "p"), "scrolledPx": full},
        {**scroll_row(True, "p"), "scrolledPx": 0},
        {**scroll_row(True, "p"), "scrolledPx": full},
    ]
    actions = clean_actions()
    actions["scroll"] = HARNESS.summarise({"scroll": rows})["scroll"]
    assert actions["scroll"]["scrolledPx"] == full, "the median still reads a full gesture"
    assert any(
        "travelled only 0px" in f and "on repetition 2" in f for f in failures(actions)
    ), failures(actions)


def test_the_verdict_rejects_a_menu_that_opened_empty_in_one_repetition() -> None:
    # An empty popover satisfies "the menu opened" and costs nothing to render, and [5, 0, 5]
    # medians to 5.
    rows = [menu_row(80.0), menu_row(100.0), menu_row(120.0)]
    rows[1]["itemsWhileOpen"] = 0
    actions = clean_actions()
    actions["menu"] = HARNESS.summarise({"menu": rows})["menu"]
    assert actions["menu"]["itemsWhileOpen"] == 5, "the median still reads a populated menu"
    assert any("no items in it on repetition(s) [2]" in f for f in failures(actions)), failures(
        actions
    )


def test_the_verdict_rejects_a_delete_whose_count_did_not_drop_in_one_repetition() -> None:
    # before and after are medianed INDEPENDENTLY, so a repetition that deleted nothing hides
    # between two that did: the medians below are 20 and 19, which reads as a clean drop, while
    # repetition 3 clicked delete and the count did not move.
    rows = [
        {"name": "delete", "ran": True, "ms": 120.0, "before": 20, "after": after}
        for after in (19, 19, 20)
    ]
    actions = clean_actions()
    actions["delete"] = HARNESS.summarise({"delete": rows})["delete"]
    assert actions["delete"]["before"] == 20 and actions["delete"]["after"] == 19, actions["delete"]
    assert any(
        "the message count did not drop on repetition 3 (20 -> 20)" in f for f in failures(actions)
    ), failures(actions)


# ── the fixture between repetitions ───────────────────────────────────


def test_the_verdict_rejects_repetitions_measured_against_different_threads() -> None:
    # The isolated delete arm reuses one page for REPEATS repetitions and each one removes
    # another assistant message from the RUNTIME's repository, which re-opening does not undo.
    # The fixture is whole cycles of one message per kind, so the three timings behind the median
    # delete three different subtree types on a shrinking thread.
    rows = [
        {
            "name": "delete",
            "ran": True,
            "ms": 120.0,
            "before": b,
            "after": b - 1,
            "fixture_messages": b,
        }
        for b in (20, 19, 18)
    ]
    actions = clean_actions()
    actions["delete"] = HARNESS.summarise({"delete": rows})["delete"]
    assert any(
        "against [20, 19, 18] messages across its repetitions" in f for f in failures(actions)
    ), failures(actions)


def test_a_restored_fixture_is_not_a_failure() -> None:
    # Expected green in both directions: a no-regression guard so the check above is known to be
    # firing on the drift rather than on the field existing at all.
    rows = [
        {
            "name": "delete",
            "ran": True,
            "ms": 120.0,
            "before": 20,
            "after": 19,
            "fixture_messages": 20,
        }
        for _ in range(3)
    ]
    actions = clean_actions()
    actions["delete"] = HARNESS.summarise({"delete": rows})["delete"]
    assert failures(actions) == []


# ── the predecessor probe ─────────────────────────────────────────────
#
# scroll_predecessor_probe.py drives the harness's OWN action scripts as predecessors to an
# otherwise identical scroll. It shares both defects above: its repetition loop reuses one page
# per arm, and it published whatever the predecessor returned without ever reading it.


def _load_probe():
    """The probe, on the stub `_load_harness` already installed for the harness it imports."""
    import scroll_predecessor_probe
    return scroll_predecessor_probe


PROBE = _load_probe()


def test_a_predecessor_that_timed_out_fails_its_arm() -> None:
    # MENU_JS returns NORMALLY with openMs None when the menu never opened inside its timeout, so
    # nothing raises, the arm measures a scroll with no predecessor in front of it, and the row is
    # published under the label `menu`.
    timed_out = {
        "openMs": None,
        "closeMs": 40.0,
        "itemsWhileOpen": 5,
        "bodyPointerEvents": "none",
    }
    with pytest.raises(RuntimeError, match = "the menu never opened"):
        PROBE.checked("menu", timed_out)


def test_a_predecessor_whose_target_was_missing_fails_its_arm() -> None:
    # Every action script starts with `if (!element) return null`.
    with pytest.raises(RuntimeError, match = "returned null"):
        PROBE.checked("delete", None)


def test_a_delete_predecessor_that_removed_nothing_fails_its_arm() -> None:
    # `ms` can be a real number while the count did not move: DELETE_JS polls `target.isConnected`
    # on the captured node, so a message re-parented rather than removed resolves the wait.
    with pytest.raises(RuntimeError, match = "did not drop"):
        PROBE.checked("delete", {"ms": 120.0, "before": 20, "after": 20})


def test_a_keystroke_predecessor_that_never_reached_the_runtime_fails_its_arm() -> None:
    with pytest.raises(RuntimeError, match = "the composer holds"):
        PROBE.checked(
            "keystroke",
            {"median_sample_ms": 33.0, "domText": "aaaaa", "runtimeText": ""},
        )


def test_a_completed_predecessor_is_returned_unchanged() -> None:
    # Expected green in both directions: a no-regression guard, so the four above are known to be
    # rejecting the failure rather than rejecting every shape of proof.
    good = {"openMs": 100.0, "closeMs": 40.0, "itemsWhileOpen": 5}
    assert PROBE.checked("menu", good) is good
    # startedFrom/bottom are part of the jump's own report now: travelledPx is observed rather
    # than the planned full height, so a repetition that began part-way up the thread is a
    # shorter gesture and the proof rejects it. A fixture without them is unverifiable, not clean.
    assert (
        PROBE.checked(
            "jump",
            {"landedAt": 0, "travelledPx": 19000, "startedFrom": 19000, "bottom": 19000},
        )
        is not None
    )
    assert PROBE.checked("reopen", {"ms": 500.0, "closedMs": 30.0, "before": 20, "after": 20})


def test_the_probe_fails_an_arm_whose_thread_shrank_between_repetitions() -> None:
    # `delete` and `delete_reopen_keystroke` remove a message from the runtime's repository on
    # every pass, and the fixture is whole cycles of one message per kind, so repetitions 2 to 4
    # scroll progressively smaller threads missing different content kinds -- against a `nothing`
    # control that still holds the whole fixture.
    assert PROBE.fixture_drift([20, 19, 18, 17]) is not None
    assert "was not restored" in PROBE.fixture_drift([20, 19, 18, 17])


def test_a_probe_arm_whose_thread_held_still_is_not_a_failure() -> None:
    # Expected green in both directions.
    assert PROBE.fixture_drift([20, 20, 20, 20]) is None
    assert PROBE.fixture_drift([]) is None
