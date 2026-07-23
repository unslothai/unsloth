# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for the cross-chat prompt queue scheduler.

These guards cover the concurrency and lifecycle boundaries that are easy to
regress when the chat runtime, sidebar, or composer is refactored. TypeScript
and the production bundle provide the executable compile checks; these tests
pin the scheduler-specific ownership rules.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
THREAD = (FRONTEND / "components/assistant-ui/thread.tsx").read_text(encoding = "utf-8")
SHARED_COMPOSER = (FRONTEND / "features/chat/shared-composer.tsx").read_text(encoding = "utf-8")
APP_SIDEBAR = (FRONTEND / "components/app-sidebar.tsx").read_text(encoding = "utf-8")
RUNTIME_STORE = (FRONTEND / "features/chat/stores/chat-runtime-store.ts").read_text(
    encoding = "utf-8"
)
CHAT_ADAPTER = (FRONTEND / "features/chat/api/chat-adapter.ts").read_text(encoding = "utf-8")
QUEUE_BOUNDARY = (FRONTEND / "features/chat/utils/prompt-queue-boundary.ts").read_text(
    encoding = "utf-8"
)
CLEAR_ALL_CHATS = (FRONTEND / "features/chat/utils/clear-all-chats.ts").read_text(
    encoding = "utf-8"
)
SIDEBAR_ITEMS = (FRONTEND / "features/chat/hooks/use-chat-sidebar-items.ts").read_text(
    encoding = "utf-8"
)


def _between(source: str, start: str, end: str) -> str:
    assert start in source, f"missing start marker: {start}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing end marker after {start}: {end}"
    return tail.split(end, 1)[0]


def test_scheduler_reserves_global_capacity_before_async_dispatch():
    pump = _between(
        THREAD,
        "function pumpPromptQueues()",
        "async function dispatchQueuedPrompt(",
    )
    append = _between(
        THREAD,
        "function appendQueuedPrompt(",
        "async function targetHasIndexingDocuments(",
    )

    assert "const PROMPT_QUEUE_GLOBAL_CONCURRENCY = 1;" in THREAD
    assert "while (promptQueueHasCapacity())" in pump
    assert pump.index("promptQueueDispatchingRunIds.add(run.id)") < pump.index(
        "dispatchQueuedPrompt(run, item, run.generation)"
    )
    assert append.index("promptQueueActiveRunIds.add(run.id)") < append.index(
        "item.target.append(item.prompt)"
    )
    assert "promptQueueActiveRunIds.size" in THREAD
    assert "promptQueueDispatchingRunIds.size" in THREAD
    assert "getPreStreamRunReservationCount()" in THREAD


def test_scheduler_pumps_when_any_generation_releases_capacity():
    subscription = _between(
        THREAD,
        "function ensurePromptQueueSubscription()",
        "function startPromptQueue(",
    )
    assert "nextRunningCount < previousRunningCount" in subscription
    assert "hasReadyPromptQueueRun()" in subscription
    assert "requestPromptQueuePump();" in subscription


def test_background_target_keeps_runtime_ownership_without_reselecting_chat():
    target = _between(
        THREAD,
        "const createPromptQueueTarget = useCallback(",
        "const dismissWaitToast",
    )
    assert "initialRunningThreadIds" in target
    assert "runtime.threads.getById(id)" in target
    assert "incognitoAtQueueStart" in target
    assert "markThreadIncognito(id)" in target
    assert "setActiveThreadId" not in target


def test_background_indexing_check_does_not_freeze_after_unmount():
    indexing = _between(
        THREAD,
        "async function targetHasIndexingDocuments(",
        "function getActivePromptQueueItem(",
    )
    target = _between(
        THREAD,
        "const createPromptQueueTarget = useCallback(",
        "const dismissWaitToast",
    )
    assert "promptQueueTargetMountedRef.current && indexingActiveRef.current" in target
    assert "usesThreadDocumentsAtQueueStart" in target
    assert "listThreadDocuments(threadId)" in indexing
    assert "useChatRuntimeStore.getState()" not in indexing


def test_regular_and_compare_sends_share_the_queue_capacity_boundary():
    submit = _between(
        THREAD,
        "const handleSubmit = useCallback(",
        "const stopQueue = useCallback(",
    )
    assert "anyPromptQueueRunning" in submit
    assert "promptQueueAtCapacity" in submit
    assert "startPromptQueue(" in submit
    assert "compareSendAtGlobalCapacity()" in SHARED_COMPOSER
    assert "usePromptQueueUI.getState().isRunning" in SHARED_COMPOSER


def test_compare_completion_wait_is_scoped_to_its_own_thread_ids():
    compare = _between(
        SHARED_COMPOSER,
        "export function RegisterCompareHandle(",
        "export function SharedComposer(",
    )
    assert "getCompareThreadIds" in compare
    assert "threadIds.some((threadId) => runningByThreadId[threadId])" in compare
    assert "Object.values(runningByThreadId).some(Boolean)" in compare


def test_compare_wait_rejects_when_pre_stream_validation_fails():
    compare = _between(
        SHARED_COMPOSER,
        "export function RegisterCompareHandle(",
        "export function SharedComposer(",
    )
    assert "PRE_STREAM_RUN_FAILED_EVENT" in compare
    assert "getCompareThreadIds().includes(failedThreadId)" in compare
    assert "reject(error);" in compare
    assert "notifyPreStreamRunFailed(resolvedThreadId ?? null)" in CHAT_ADAPTER
    assert "notifyPromptQueueRunFailed(threadId)" in QUEUE_BOUNDARY


def test_normal_sends_reserve_capacity_until_stream_ownership_begins():
    submit = _between(
        THREAD,
        "const handleSubmit = useCallback(",
        "const stopQueue = useCallback(",
    )
    assert "tryReservePreStreamRun()" in submit
    assert "preStreamRunReservations" in QUEUE_BOUNDARY
    assert "releasePreStreamRunReservation();" in CHAT_ADAPTER
    assert "runtime.setThreadRunning(threadKey, true);" in CHAT_ADAPTER


def test_bulk_archive_and_clear_stop_prompt_queues_first():
    archive_all = _between(
        SIDEBAR_ITEMS,
        "export async function archiveAllChatItems(",
        "export async function unarchiveChatItem(",
    )
    assert "requestPromptQueueStop(toArchive.map((thread) => thread.id));" in archive_all
    assert "requestPromptQueueStop();" in CLEAR_ALL_CHATS
    assert CLEAR_ALL_CHATS.index("requestPromptQueueStop();") < CLEAR_ALL_CHATS.index(
        "clearStoredChats();"
    )


def test_cancel_and_failure_paths_release_capacity_and_resume_other_queues():
    stop = _between(
        THREAD,
        "function stopPromptQueueRun(threadIds?: string[])",
        "function stopPromptQueueRunForThreadIds(",
    )
    failed = _between(
        THREAD,
        "function handlePromptQueueRunFailed(",
        'if (typeof window !== "undefined")',
    )
    assert "activeTarget?.cancel()" in stop
    assert "requestPromptQueuePumpIfReady();" in stop
    assert "deletePromptQueueRun(failedRun);" in failed
    assert "requestPromptQueuePumpIfReady();" in failed
    assert "cancelByThreadId: Record<string, () => void>;" in RUNTIME_STORE


def test_sidebar_distinguishes_running_queues_from_completed_background_chats():
    assert "const queueByThreadId = usePromptQueueUI((s) => s.byThreadId);" in APP_SIDEBAR
    assert "previousRunningByThreadIdRef" in APP_SIDEBAR
    assert "hasQueueActivity" in APP_SIDEBAR
    assert "hasUnreadActivity" in APP_SIDEBAR
    assert "clearChatNotifications(item)" in APP_SIDEBAR
