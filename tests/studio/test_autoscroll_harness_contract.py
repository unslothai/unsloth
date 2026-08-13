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
    # nothing.
    main = verdict("playwright_research_freeze.py")
    assert 'stream["raf_per_second"]' in main
    assert 'stream["activities"]' in main
