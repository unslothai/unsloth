# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Composer stop pauses the queue; delete/archive still delete it."""

from pathlib import Path

THREAD = (
    Path(__file__).resolve().parents[2]
    / "studio/frontend/src/components/assistant-ui/thread.tsx"
).read_text(encoding = "utf-8")


def _between(source: str, start: str, end: str) -> str:
    assert start in source, f"missing start marker: {start}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing end marker after {start}: {end}"
    return tail.split(end, 1)[0]


def test_composer_stop_pauses_without_permanently_cancelling_the_target():
    pause = _between(
        THREAD,
        "function pausePromptQueueRun(threadIds?: string[])",
        "function resumePromptQueueRun(threadIds?: string[])",
    )
    assert "userStopTargetCancelMode(plan)" in pause
    assert "cancelActiveRun()" in pause
    assert "pausePromptQueueRun(promptQueueThreadIds)" in THREAD
    assert 'aria-label="Resume queue"' in THREAD
    stop = _between(
        THREAD,
        "function stopPromptQueueRun(threadIds?: string[])",
        "function stopPromptQueueRunForThreadIds(threadIds: string[])",
    )
    assert "planUserPromptQueueStop(" not in stop
    assert "deletePromptQueueRun(run);" in stop
    listener = _between(
        THREAD,
        "window.addEventListener(PROMPT_QUEUE_STOP_EVENT",
        "window.addEventListener(PROMPT_QUEUE_RUN_FAILED_EVENT",
    )
    assert "stopPromptQueueRunForThreadIds(threadIds)" in listener
    assert "pausePromptQueueRun(" not in listener
