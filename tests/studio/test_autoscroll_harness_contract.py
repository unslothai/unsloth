# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two #8483 harnesses must read every guard they record.

Both files measure in a browser and then decide pass/fail in `main()`. A metric that is recorded
but never compared is how a harness goes false-green: it keeps reporting the number that would
have caught the regression while exiting 0. Three shipped that way already (an unasserted rAF
budget, an unasserted click count, and the two guards below), so the rule is pinned here rather
than left to review.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"


def source(name: str) -> str:
    return (STUDIO_TESTS / name).read_text(encoding = "utf-8")


def verdict(name: str) -> str:
    """Everything from `def main()` on: the only place a metric turns into an exit code."""
    text = source(name)
    start = text.index("def main()")
    return text[start:]


def test_chat_autoscroll_asserts_the_detached_stream_actually_grew() -> None:
    # `stillDetached` says streaming did not re-pin the reader. It says nothing at all if the
    # streamed tokens added no height, which the harness measures as `grewWhileDetached`.
    main = verdict("playwright_chat_autoscroll.py")
    assert 'intent["stillDetached"]' in main
    assert 'intent["grewWhileDetached"]' in main


def test_research_freeze_asserts_the_dialog_took_the_modal_layer() -> None:
    # The stranded-`pointer-events: none` checks are the reported symptom (an unclickable
    # window). They only test anything if the dialog put the body on the modal layer first.
    main = verdict("playwright_research_freeze.py")
    assert 'modal["body_pointer_events_after_approve"]' in main
    assert 'modal["body_pointer_events_while_open"]' in main


def test_research_freeze_asserts_the_stream_had_something_to_follow() -> None:
    # An empty activity list runs no follow loop, so it clears the frame budgets by measuring
    # nothing. `seed()` leaves activities behind, so the count has to be compared against the
    # pre-stream baseline rather than against zero.
    main = verdict("playwright_research_freeze.py")
    assert 'stream["raf_per_second"]' in main
    assert 'stream["activities"]' in main
    assert 'stream["activities_before"]' in main


def test_research_freeze_asserts_the_report_stall_and_its_own_probe() -> None:
    # The stall budget is only a budget if a stall of zero fails too: no second sample means
    # the probe measured nothing and the comparison below it passes on any tree.
    main = verdict("playwright_research_freeze.py")
    assert 'results["report"]["main_thread_stall_ms"] > MAIN_THREAD_STALL_BUDGET_MS' in main
    assert 'results["report"]["main_thread_stall_ms"] <= 0' in main


def test_research_freeze_keeps_a_hit_tested_click_in_the_report_phase() -> None:
    # A synthetic element.click() bypasses hit testing, so it lands even with
    # `body { pointer-events: none }` stranded, which is the freeze being tested. Only a real
    # Playwright click fails that tree, so the verdict must read the actionability result and
    # not just the handler's own counter.
    source_text = source("playwright_research_freeze.py")
    assert 'page.click(\'[data-smoke="click-probe"]\'' in source_text
    main = verdict("playwright_research_freeze.py")
    assert 'results["report"]["click_landed"]' in main
    assert 'results["report"]["clicks_registered"]' in main


def test_harnesses_own_their_dev_server() -> None:
    # A server started beside the harness (a backgrounded `npm run dev` in a CI step) leaves the
    # node child alive when the wrapper is killed, which strands the port and the step's stdout.
    # Each harness starts and stops its own instead, through the shared helpers.
    for name in (
        "playwright_chat_autoscroll.py",
        "playwright_research_freeze.py",
        "playwright_strip_ansi_smoke.py",
    ):
        text = source(name)
        assert "start_vite" in text, f"{name} does not start its own server"
        assert "stop_process" in text, f"{name} never tears its server down"
        # Status alone is satisfied by vite's SPA fallback, which answers 200 with index.html
        # for a smoke page that no longer exists.
        assert "wait_for_smoke_page" in text, f"{name} gates on status rather than on content"
