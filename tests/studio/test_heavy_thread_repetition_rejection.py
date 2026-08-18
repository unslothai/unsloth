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
