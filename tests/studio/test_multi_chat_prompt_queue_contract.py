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
RUNTIME_PROVIDER = (FRONTEND / "features/chat/runtime-provider.tsx").read_text(encoding = "utf-8")
QUEUE_BOUNDARY = (FRONTEND / "features/chat/utils/prompt-queue-boundary.ts").read_text(
    encoding = "utf-8"
)
CLEAR_ALL_CHATS = (FRONTEND / "features/chat/utils/clear-all-chats.ts").read_text(encoding = "utf-8")
SIDEBAR_ITEMS = (FRONTEND / "features/chat/hooks/use-chat-sidebar-items.ts").read_text(
    encoding = "utf-8"
)
CHAT_PAGE = (FRONTEND / "features/chat/chat-page.tsx").read_text(encoding = "utf-8")
QUEUED_SETTINGS = (FRONTEND / "features/chat/utils/queued-chat-run-settings.ts").read_text(
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
    assert "onSendClick={handleSubmit}" in THREAD
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
    assert "registerPreStreamRun(options.unstable_threadId ?? null)" in RUNTIME_PROVIDER
    assert "releasePreStreamRunForThread(threadKey);" in CHAT_ADAPTER
    assert "runtime.setThreadRunning(threadKey, true);" in CHAT_ADAPTER
    assert "sendReservedComposer()" in submit
    assert "notifyPreStreamRunFailed(referenceThreadId);" in THREAD
    assert "class PreStreamAwareAttachmentAdapter" in RUNTIME_PROVIDER
    attachment_adapter = _between(
        RUNTIME_PROVIDER,
        "class PreStreamAwareAttachmentAdapter",
        "function useStudioRuntimeAdapters(",
    )
    assert "return await this.delegate.send(attachment);" in attachment_adapter
    assert "getPreStreamRunReservationCount() > 0" in attachment_adapter
    assert "notifyPreStreamRunFailed();" in attachment_adapter
    assert "new PreStreamAwareAttachmentAdapter(" in RUNTIME_PROVIDER
    assert "Promise.resolve(aui.composer().send())" not in THREAD


def test_retry_edit_and_compare_preflights_reserve_global_capacity():
    assert THREAD.count("reserveInteractiveRun(event)") >= 3
    assert "if (!reserveInteractiveRun()) return;" in THREAD
    assert "if (!tryReservePreStreamRun())" in SHARED_COMPOSER
    assert "compareReservationPending = true;" in SHARED_COMPOSER
    assert "notifyPreStreamRunFailed();" in SHARED_COMPOSER
    reserve = _between(
        THREAD,
        "function reserveInteractiveRun(",
        "function isPromptQueueRunReadyToDispatch(",
    )
    assert "!usePromptQueueUI.getState().isRunning" in reserve


def test_background_queue_snapshots_settings_and_blocks_model_changes():
    target = _between(
        THREAD,
        "const createPromptQueueTarget = useCallback(",
        "const dismissWaitToast",
    )
    assert "snapshotQueuedChatRunSettings(chatStateAtQueueStart)" in target
    assert "registerQueuedChatRunSettings(" in target
    assert "consumeQueuedChatRunSettings(resolvedThreadId)" in CHAT_ADAPTER
    assert "params: { ...state.params }" in QUEUED_SETTINGS
    assert '"reasoningEnabled"' in QUEUED_SETTINGS
    assert '"reasoningEffort"' in QUEUED_SETTINGS
    assert '"preserveThinking"' in QUEUED_SETTINGS
    assert "pendingSettingsIds" in target
    assert "discardQueuedChatRunSettings(settingsId)" in target
    assert "discardQueuedChatRunSettingsForThread(threadId);" in THREAD
    assert "entry.threadIds.has(threadId)" in QUEUED_SETTINGS
    assert "queuedRunSettings.params.checkpoint" in CHAT_ADAPTER
    assert "? queuedRunSettings.params.checkpoint" in CHAT_ADAPTER
    assert ": liveRuntime" in CHAT_ADAPTER
    assert "usePromptQueueUI.getState().isRunning" in CHAT_PAGE
    assert "Object.values(runtime.runningByThreadId).some(Boolean)" in CHAT_PAGE
    assert "getPreStreamRunReservationCount() > 0" in CHAT_PAGE


def test_bulk_archive_and_clear_stop_prompt_queues_first():
    archive_all = _between(
        SIDEBAR_ITEMS,
        "export async function archiveAllChatItems(",
        "export async function unarchiveChatItem(",
    )
    assert "requestPromptQueueStop(toArchive.map((thread) => thread.id));" in archive_all
    assert "isPreStreamRunActive(threadId)" in SIDEBAR_ITEMS
    assert "cancelByThreadId[threadId]?.();" in CLEAR_ALL_CHATS
    assert "getPreStreamRunThreadIds()" in CLEAR_ALL_CHATS
    assert "requestPromptQueueStop();" in CLEAR_ALL_CHATS
    assert CLEAR_ALL_CHATS.index("cancelByThreadId[threadId]?.();") < (
        CLEAR_ALL_CHATS.index("requestPromptQueueStop();")
    )
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
    assert failed.index("requestPromptQueuePumpIfReady();") > failed.index("if (threadId)")
    assert "cancelByThreadId: Record<string, () => void>;" in RUNTIME_STORE
    assert "notifyPreStreamRunFailed(options.unstable_threadId ?? null)" in RUNTIME_PROVIDER
    set_running = _between(
        RUNTIME_STORE,
        "setThreadRunning: (threadId, running) =>",
        "registerThreadCancel: (threadId, cancel) =>",
    )
    assert "delete nextCancel[threadId]" not in set_running
    assert "registrar cleanup removes it" in set_running


def test_persisted_new_chat_accepts_its_promoted_remote_id():
    assert (
        RUNTIME_PROVIDER.count(
            "visibleThreadId === localThreadId ||\n                visibleThreadId === remoteId"
        )
        >= 2
    )


def test_sidebar_distinguishes_running_queues_from_completed_background_chats():
    assert "const queueByThreadId = usePromptQueueUI((s) => s.byThreadId);" in APP_SIDEBAR
    assert "previousRunningByThreadIdRef" in APP_SIDEBAR
    assert "hasQueueActivity" in APP_SIDEBAR
    assert "hasUnreadActivity" in APP_SIDEBAR
    assert "clearChatNotifications(item)" in APP_SIDEBAR
