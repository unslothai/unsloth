# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop generation must keep prompts that have not been sent yet.

https://github.com/unslothai/unsloth/issues/10428
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
THREAD = (FRONTEND / "components/assistant-ui/thread.tsx").read_text(encoding = "utf-8")
PLANNER_PATH = FRONTEND / "features/chat/utils/prompt-queue-user-stop.ts"
BARREL = (FRONTEND / "features/chat/index.ts").read_text(encoding = "utf-8")


def _between(source: str, start: str, end: str) -> str:
    assert start in source, f"missing start marker: {start}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing end marker after {start}: {end}"
    return tail.split(end, 1)[0]


def test_user_stop_plans_with_pending_items_kept():
    assert PLANNER_PATH.is_file()
    planner = PLANNER_PATH.read_text(encoding = "utf-8")
    assert "export function planUserPromptQueueStop(" in planner
    assert 'from "./utils/prompt-queue-user-stop"' in BARREL
    stop = _between(
        THREAD,
        "function stopPromptQueueRun(threadIds?: string[])",
        "function stopPromptQueueRunForThreadIds(threadIds: string[])",
    )
    assert "planUserPromptQueueStop(" in stop
    assert stop.index("planUserPromptQueueStop(") < stop.index(
        "deletePromptQueueRun(run);"
    )
    assert "run.paused = plan.pause" in stop
    assert "run.generation += 1" in stop


def test_paused_queue_does_not_dispatch_or_advance():
    ready = _between(
        THREAD,
        "function isPromptQueueRunReadyToDispatch(run: PromptQueueRun)",
        "function getNextReadyPromptQueueRun()",
    )
    assert "!run.paused" in ready
    run_state = _between(
        THREAD,
        "function handlePromptQueueRunState(",
        "function ensurePromptQueueSubscription()",
    )
    assert run_state.index("if (run.paused)") < run_state.index(
        "advancePromptQueue(run);"
    )
    retained = _between(
        THREAD,
        "function retainPendingPromptQueueItemsAfterFailure(run: PromptQueueRun)",
        "function cancelPendingPromptQueueFactoriesForStop<",
    )
    assert retained.index("if (run.paused)") < retained.index(
        "run.items.splice(activeIndex, 1);"
    )


def test_start_unpauses_an_existing_run():
    start = _between(
        THREAD,
        "function startPromptQueue(",
        "function getPromptQueueRunsForThreadIds(threadIds?: string[])",
    )
    assert "existingRun.paused = false" in start
    assert "existingRun?.paused" in start
    assert "paused: false" in start
