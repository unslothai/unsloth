# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

THREAD = (
    Path(__file__).resolve().parents[2] / "studio/frontend/src/components/assistant-ui/thread.tsx"
).read_text(encoding = "utf-8")


def test_composer_stop_pauses_and_delete_still_deletes():
    assert "pausePromptQueueRun(promptQueueThreadIds)" in THREAD
    assert "cancelActiveRun()" in THREAD
    assert 'aria-label="Resume queue"' in THREAD
    stop = THREAD.split("function stopPromptQueueRun(threadIds?: string[])", 1)[1]
    stop = stop.split("function stopPromptQueueRunForThreadIds", 1)[0]
    assert "planUserPromptQueueStop(" not in stop
    listener = THREAD.split("window.addEventListener(PROMPT_QUEUE_STOP_EVENT", 1)[1]
    listener = listener.split("window.addEventListener(PROMPT_QUEUE_RUN_FAILED_EVENT", 1)[0]
    assert "stopPromptQueueRunForThreadIds(threadIds)" in listener
    assert "pausePromptQueueRun(" not in listener
