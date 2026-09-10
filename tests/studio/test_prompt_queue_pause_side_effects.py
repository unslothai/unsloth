# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A paused run keeps its PromptQueueUIEntry, so every reader of "an entry exists"
had to learn about `paused`. These pin those places and the two dispatch races."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THREAD = (ROOT / "studio/frontend/src/components/assistant-ui/thread.tsx").read_text(
    encoding = "utf-8"
)
SIDEBAR = (ROOT / "studio/frontend/src/components/app-sidebar.tsx").read_text(encoding = "utf-8")
CONFIRM = (
    ROOT / "studio/frontend/src/features/chat/utils/confirm-stop-running-chats.ts"
).read_text(encoding = "utf-8")


def test_the_sidebar_work_spinner_ignores_a_paused_queue():
    block = SIDEBAR.split("const hasQueuedActivity", 1)[1]
    block = block.split("const showQueuedActivity", 1)[0]
    assert "entry.paused" in block


def test_a_paused_queue_is_not_a_running_chat_in_the_stop_dialog():
    loop = CONFIRM.split("for (const threadId of promptQueueThreadIds)", 1)[1]
    loop = loop.split("for (const aliases of aliasesByQueuedRun.values())", 1)[0]
    assert "entry.paused" in loop


def test_resume_clears_the_stale_running_edge_so_it_cannot_skip_a_prompt():
    resume = THREAD.split("function resumePromptQueueRun(threadIds?: string[])", 1)[1]
    resume = resume.split("function stopPromptQueueRun", 1)[0]
    assert "run.prevStoreRunning = false" in resume
    assert "run.waitingForTargetIdle = false" in resume
    assert "clearPromptQueueRetryTimer(run)" in resume


def test_a_superseded_dispatch_does_not_release_the_live_dispatch_flag():
    pump = THREAD.split("function pumpPromptQueues()", 1)[1]
    pump = pump.split("async function dispatchQueuedPrompt(", 1)[0]
    assert "const dispatchGeneration = run.generation" in pump
    assert "dispatchGeneration !== run.generation" in pump


def test_pause_does_not_rewind_past_a_prompt_the_run_already_moved_beyond():
    pause = THREAD.split("function pausePromptQueueRun(threadIds?: string[])", 1)[1]
    pause = pause.split("function resumePromptQueueRun", 1)[0]
    assert "resumeFrom" in pause
    assert "index >= searchFrom" in pause
