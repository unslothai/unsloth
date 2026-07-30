# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for independent per-chat prompt queues."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
THREAD = (FRONTEND / "components/assistant-ui/thread.tsx").read_text(encoding = "utf-8")
APP_SIDEBAR = (FRONTEND / "components/app-sidebar.tsx").read_text(encoding = "utf-8")
CHAT_ADAPTER = (FRONTEND / "features/chat/api/chat-adapter.ts").read_text(encoding = "utf-8")
MODEL_RUNTIME = (FRONTEND / "features/chat/hooks/use-chat-model-runtime.ts").read_text(
    encoding = "utf-8"
)
CONFIRM_MODEL_SWAP = (FRONTEND / "features/chat/utils/confirm-stop-running-chats.ts").read_text(
    encoding = "utf-8"
)
RUNTIME_PROVIDER = (FRONTEND / "features/chat/runtime-provider.tsx").read_text(encoding = "utf-8")
QUEUE_BOUNDARY = (FRONTEND / "features/chat/utils/prompt-queue-boundary.ts").read_text(
    encoding = "utf-8"
)
QUEUED_SETTINGS = (FRONTEND / "features/chat/utils/queued-chat-run-settings.ts").read_text(
    encoding = "utf-8"
)
SIDEBAR_ITEMS = (FRONTEND / "features/chat/hooks/use-chat-sidebar-items.ts").read_text(
    encoding = "utf-8"
)
CLEAR_ALL_CHATS = (FRONTEND / "features/chat/utils/clear-all-chats.ts").read_text(encoding = "utf-8")


def _between(source: str, start: str, end: str) -> str:
    assert start in source, f"missing start marker: {start}"
    tail = source.split(start, 1)[1]
    assert end in tail, f"missing end marker after {start}: {end}"
    return tail.split(end, 1)[0]


def test_scheduler_dispatches_each_ready_chat_without_a_frontend_global_cap():
    pump = _between(
        THREAD,
        "function pumpPromptQueues()",
        "async function dispatchQueuedPrompt(",
    )
    assert "while (true)" in pump
    assert pump.index("promptQueueDispatchingRunIds.add(run.id)") < pump.index(
        "dispatchQueuedPrompt(run, item, run.generation)"
    )
    assert "PROMPT_QUEUE_GLOBAL_CONCURRENCY" not in THREAD
    assert "promptQueueHasCapacity" not in THREAD
    assert "preStreamRunReservations" not in QUEUE_BOUNDARY


def test_each_chat_queue_stays_sequential_and_targets_its_background_runtime():
    target = _between(
        THREAD,
        "const createPromptQueueTarget = useCallback(",
        "const dismissWaitToast",
    )
    state_handler = _between(
        THREAD,
        "function handlePromptQueueRunState(",
        "function ensurePromptQueueSubscription(",
    )
    assert "runtime.threads.getById(id)" in target
    assert "setActiveThreadId" not in target
    assert "return runningIds.length > 0" not in THREAD
    assert "isPromptQueueRunTargetRunning(run" in state_handler
    assert "advancePromptQueue(run)" in state_handler
    assert "promptQueueActiveRunIds.has(run.id)" in THREAD


def test_switching_or_starting_a_chat_does_not_stop_an_existing_queue():
    switchers = _between(
        RUNTIME_PROVIDER,
        "function ThreadAutoSwitch(",
        "function ActiveThreadSync(",
    )
    assert "requestPromptQueueStop" not in switchers
    assert "switchToThread(threadId)" in switchers
    assert "switchToNewThread()" in switchers


def test_composer_only_queues_behind_the_current_chat():
    submit = _between(
        THREAD,
        "const handleSubmit = useCallback(",
        "const stopQueue = useCallback(",
    )
    assert "aui.thread().getState().isRunning" in submit
    assert "usePromptQueueUI.getState()" in submit
    assert "if (liveThreadIsRunning || livePromptQueueActive)" in submit
    assert "startPromptQueue(" in submit
    assert "anyPromptQueueRunning" not in submit
    assert "promptQueueAtCapacity" not in submit
    assert "tryReservePreStreamRun" not in submit


def test_queued_settings_are_thread_scoped_without_cross_chat_fallback():
    target = _between(
        THREAD,
        "const createPromptQueueTarget = useCallback(",
        "const dismissWaitToast",
    )
    assert "snapshotQueuedChatRunSettings(chatStateAtQueueStart)" in target
    assert "registerQueuedChatRunSettings(" in target
    assert "consumeQueuedChatRunSettings(resolvedThreadId)" in CHAT_ADAPTER
    assert '"deepResearchEnabled"' in QUEUED_SETTINGS
    assert CHAT_ADAPTER.index(
        "consumeQueuedChatRunSettings(resolvedThreadId)"
    ) < CHAT_ADAPTER.index("if (runtime.deepResearchEnabled && threadAlreadyResearched)")
    assert "pendingSettings.length === 1" not in QUEUED_SETTINGS
    assert "entry.threadIds.has(threadId)" in QUEUED_SETTINGS
    assert "notifyPromptQueueRunFailed(resolvedThreadId ?? null)" in CHAT_ADAPTER
    assert "usesLocalModel:" in target
    assert "usePromptQueueUI.getState().byThreadId" in CONFIRM_MODEL_SWAP
    assert "getLocalPromptQueueThreadIds" in CONFIRM_MODEL_SWAP
    assert "promptQueueThreadIds" in MODEL_RUNTIME
    assert "const promptQueueThreadIds = getLocalPromptQueueThreadIds()" in MODEL_RUNTIME
    assert "requestPromptQueueStop(promptQueueThreadIds)" in MODEL_RUNTIME
    assert "function promptQueueRunUsesLocalModel(run: PromptQueueRun)" in THREAD
    assert ".slice(Math.max(run.index, 0))" in THREAD
    assert ".some((item) => item.target.usesLocalModel)" in THREAD
    assert "local: promptQueueRunUsesLocalModel(run)" in THREAD


def test_stop_delete_archive_and_clear_are_thread_scoped():
    stop_listener = _between(
        THREAD,
        "window.addEventListener(PROMPT_QUEUE_STOP_EVENT",
        "window.addEventListener(PROMPT_QUEUE_RUN_FAILED_EVENT",
    )
    assert "stopPromptQueueRunForThreadIds(threadIds)" in stop_listener
    assert "requestPromptQueueStop(toArchive.map((thread) => thread.id));" in SIDEBAR_ITEMS
    assert "requestPromptQueueStop(threadIds);" in SIDEBAR_ITEMS
    assert "requestPromptQueueStop();" in CLEAR_ALL_CHATS


def test_sidebar_exposes_queue_activity_for_each_thread():
    assert "const queueByThreadId = usePromptQueueUI((s) => s.byThreadId);" in APP_SIDEBAR
    assert "hasQueuedActivity" in APP_SIDEBAR
    assert "showWorkSpinner" in APP_SIDEBAR
    assert "{showWorkSpinner && (" in APP_SIDEBAR
    assert "hasUnreadActivity" in APP_SIDEBAR
    assert "clearChatNotifications(item)" in APP_SIDEBAR
