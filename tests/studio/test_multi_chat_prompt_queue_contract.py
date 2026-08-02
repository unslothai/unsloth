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
CHAT_RUNTIME_STORE = (FRONTEND / "features/chat/stores/chat-runtime-store.ts").read_text(
    encoding = "utf-8"
)
CHAT_PAGE = (FRONTEND / "features/chat/chat-page.tsx").read_text(encoding = "utf-8")
SHARED_COMPOSER = (FRONTEND / "features/chat/shared-composer.tsx").read_text(encoding = "utf-8")
QUEUE_BOUNDARY = (FRONTEND / "features/chat/utils/prompt-queue-boundary.ts").read_text(
    encoding = "utf-8"
)
PRE_STREAM_RESERVATION = (
    FRONTEND / "features/chat/utils/pre-stream-run-reservation.ts"
).read_text(encoding = "utf-8")
QUEUED_SETTINGS = (FRONTEND / "features/chat/utils/queued-chat-run-settings.ts").read_text(
    encoding = "utf-8"
)
SIDEBAR_ITEMS = (FRONTEND / "features/chat/hooks/use-chat-sidebar-items.ts").read_text(
    encoding = "utf-8"
)
CLEAR_ALL_CHATS = (FRONTEND / "features/chat/utils/clear-all-chats.ts").read_text(encoding = "utf-8")
STOP_CHAT_THREAD = (FRONTEND / "features/chat/utils/stop-chat-thread.ts").read_text(
    encoding = "utf-8"
)


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
    assert "const reservations = new Map<symbol" in PRE_STREAM_RESERVATION
    assert "reservationByThreadId" in PRE_STREAM_RESERVATION
    assert "let preStreamRunReservations = 0" not in PRE_STREAM_RESERVATION


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
    assert "Boolean(getActivePromptQueueItem(run)?.dispatched)" in THREAD
    append = _between(
        THREAD,
        "function appendQueuedPrompt(",
        "async function targetHasIndexingDocuments(",
    )
    assert "schedulePromptQueueTargetStatePoll(run)" in append
    indexing_probe = _between(
        THREAD,
        "async function targetHasIndexingDocuments(",
        "function getActivePromptQueueItem(",
    )
    assert "const documents = await listThreadDocuments(threadId)" in indexing_probe
    assert "catch {\n    // A failed status probe" in indexing_probe
    assert "return true;" in indexing_probe


def test_saved_queues_survive_navigation_but_abandoned_temporary_queues_stop():
    saved_switch = _between(
        RUNTIME_PROVIDER,
        "function ThreadAutoSwitch(",
        "function ThreadNewChatSwitch(",
    )
    temporary_switch = _between(
        RUNTIME_PROVIDER,
        "function ThreadNewChatSwitch(",
        "function ActiveThreadSync(",
    )
    assert "requestPromptQueueStop" not in saved_switch
    assert "requestTemporaryPromptQueueStop()" in saved_switch
    assert "switchToThread(threadId)" in saved_switch
    assert "requestTemporaryPromptQueueStop()" in temporary_switch
    assert "switchToNewThread()" in temporary_switch

    temporary_toggle = _between(
        CHAT_PAGE,
        "const toggleIncognito = useCallback(",
        "const hydratePersistedSettings",
    )
    assert "if (wasIncognito)" in temporary_toggle
    assert "requestTemporaryPromptQueueStop()" in temporary_toggle
    assert temporary_toggle.index("requestTemporaryPromptQueueStop()") < temporary_toggle.index(
        "if (onEmptyScratchChat) return"
    )
    cancel_registrar = _between(
        RUNTIME_PROVIDER,
        "function CancelRegistrar()",
        "function ThreadBackendAutosave(",
    )
    assert "thread?.getState().isRunning" in cancel_registrar
    assert "unsubscribe = thread.subscribe(" in cancel_registrar
    assert "threadListItem.remoteId" in cancel_registrar
    assert "registerThreadCancel(threadId, cancel)" in cancel_registrar
    assert "clearThreadCancel(threadId, cancel)" in cancel_registrar
    assert cancel_registrar.index("thread.subscribe(") < cancel_registrar.rindex(
        "clearThreadCancel(threadId, cancel)"
    )
    assert "state.cancelByThreadId[threadId] !== cancel" in CHAT_RUNTIME_STORE


def test_composer_only_queues_behind_the_current_chat():
    submit = _between(
        THREAD,
        "const handleSubmit = useCallback(",
        "const stopQueue = useCallback(",
    )
    assert "aui.thread().getState().isRunning" in submit
    assert "usePromptQueueUI.getState()" in submit
    assert "livePreStreamRunActive" in submit
    assert "liveThreadIsRunning || livePreStreamRunActive" in submit
    assert "startHydratedPromptQueue(" in submit
    assert "aui.composer().getState().text.trim() !== queuedPrompt" in submit
    assert "promptQueueStartPendingRef.current" in THREAD
    assert "promptQueueStartPendingRef.current.has(reservationKey)" in THREAD
    assert "promptQueueStartPendingRef.current.delete(reservationKey)" in THREAD
    assert "promptQueueStartPendingRef.current.set(reservationKey, reservation)" in THREAD
    assert "temporary: useChatRuntimeStore.getState().incognito" in THREAD
    assert "localPromptQueueModelBoundary.capture()" in THREAD
    assert "shouldAbortPendingQueueForModelBoundary" in THREAD
    assert "queuedSettingsEpoch:" in THREAD
    assert "shouldAbortPendingQueueForSettingsChange" in THREAD
    assert "capturedEpoch: reservation.queuedSettingsEpoch" in THREAD
    assert "currentEpoch: currentQueueSettings.queuedSettingsEpoch" in THREAD
    assert "capturedTemporary: reservation.temporary" in THREAD
    assert "currentTemporary: currentQueueSettings.incognito" in THREAD
    assert "!settingsInvalidated" in THREAD
    assert "reservation.cancelled = true" in THREAD
    assert "temporaryOnly && !reservation.temporary" in THREAD
    assert "onAborted?.()" in THREAD
    assert 'toast.info("Saved list was not queued"' in THREAD
    assert ".finally(() =>" in THREAD
    assert "anyPromptQueueRunning" not in submit
    assert "promptQueueAtCapacity" not in submit
    assert "sendReservedComposer();" in submit
    assert "reservePreStreamRun(preStreamThreadIds)" in THREAD
    assert "adoptPreStreamRunReservation(token, preStreamThreadIds)" in THREAD
    assert "hasPreStreamRunReservation(getQueueThreadIds())" in THREAD
    assert "releaseCurrentPreStreamRun();" in CHAT_ADAPTER
    assert "releasePreStreamRunReservation(reservationToken)" in CHAT_ADAPTER
    assert "class PreStreamAwareAttachmentAdapter" in RUNTIME_PROVIDER


def test_queued_settings_are_thread_scoped_without_cross_chat_fallback():
    target = _between(
        THREAD,
        "const createPromptQueueTarget = useCallback(",
        "const dismissWaitToast",
    )
    assert "await useChatRuntimeStore.getState().hydratePersistedSettings()" in target
    assert target.index(
        "await useChatRuntimeStore.getState().hydratePersistedSettings()"
    ) < target.index("snapshotQueuedChatRunSettings(chatStateAtQueueStart)")
    assert "if (!promptQueueTargetMountedRef.current)" in target
    assert "const currentState = aui.threadListItem().getState()" in target
    assert "initialRunningThreadIds.includes(id)" in target
    assert "snapshotQueuedChatRunSettings(chatStateAtQueueStart)" in target
    assert "registerQueuedChatRunSettings(" in target
    assert "params: { ...runSettingsAtQueueStart.params }" in target
    assert "runSettingsAtQueueStart.deepResearchEnabled = false" in target
    assert target.index("const appendResult = thread.append(") < target.index(
        "runSettingsAtQueueStart.deepResearchEnabled = false"
    )
    assert "void (appendResult as Promise<void>).catch(() => undefined)" in target
    assert "function consumePromptQueueDeepResearch(" in THREAD
    assert "!item.target.usesDeepResearch" in THREAD
    assert ".then(() => consumePromptQueueDeepResearch(run, item))" in THREAD
    assert "usesDeepResearch: runSettingsAtQueueStart.deepResearchEnabled" in target
    assert "if (existingRun.deepResearchConsumed)" in THREAD
    assert "addQueuedChatRunSettingsThreadIds(settingsId" in target
    assert ".getItemById(state.id)\n            .initialize()" in target
    assert "await updateStoredChatThread(remoteId" in target
    assert "let shouldCorrectPersistedModel: boolean | null = null" in target
    assert "shouldCorrectPersistedModel ??= !state.remoteId" in target
    assert "if (shouldCorrectPersistedModel)" in target
    assert (
        target.index("await updateStoredChatThread(remoteId")
        < target.index("shouldCorrectPersistedModel = false")
        < target.index("const appendResult = thread.append(")
    )
    assert 'modelId: runSettingsAtQueueStart.params.checkpoint ?? ""' in target
    assert target.index("await updateStoredChatThread(remoteId") < target.index(
        "const appendResult = thread.append("
    )
    assert (
        target.index("addQueuedChatRunSettingsThreadIds(settingsId")
        < target.index("syncPromptQueueUI()")
        < target.index("const appendResult = thread.append(")
    )
    assert "let cancelled = false" in target
    assert "if (cancelled || !pendingSettingsIds.has(settingsId))" in target
    assert target.index("if (cancelled || !pendingSettingsIds.has(settingsId))") < target.index(
        "const appendResult = thread.append("
    )
    assert "cancelled = true" in target
    assert "isTargetCurrentThread() &&" in target
    assert "consumeQueuedChatRunSettings(resolvedThreadId)" in CHAT_ADAPTER
    assert '"deepResearchEnabled"' in QUEUED_SETTINGS
    assert '"supportsReasoning"' in QUEUED_SETTINGS
    assert '"reasoningAlwaysOn"' in QUEUED_SETTINGS
    assert '"reasoningStyle"' in QUEUED_SETTINGS
    assert '"supportsReasoningOff"' in QUEUED_SETTINGS
    assert '"reasoningEffortLevels"' in QUEUED_SETTINGS
    assert '"supportsPreserveThinking"' in QUEUED_SETTINGS
    assert '"researchWebsitePolicy"' in QUEUED_SETTINGS
    assert CHAT_ADAPTER.index(
        "consumeQueuedChatRunSettings(resolvedThreadId)"
    ) < CHAT_ADAPTER.index("if (runtime.deepResearchEnabled && threadAlreadyResearched)")
    research = _between(
        CHAT_ADAPTER,
        "if (\n        runtime.deepResearchEnabled",
        "const sandboxSessionId",
    )
    assert "const liveRuntime = useChatRuntimeStore.getState()" in research
    assert "...queuedRunSettings" in research
    auto_load_merge = _between(
        CHAT_ADAPTER,
        "// Re-read store after auto-load / model-ready wait.",
        "const { params } = runtime",
    )
    assert "...queuedRunSettings.params" in auto_load_merge
    assert "queuedEmptyModelRuntime?.checkpoint" in auto_load_merge
    assert "liveRuntime.params.checkpoint" in auto_load_merge
    assert "liveRuntime.supportsTools" in auto_load_merge
    assert "liveRuntime.supportsReasoning" in auto_load_merge
    assert "liveRuntime.ggufContextLength" in auto_load_merge
    assert "isExternalModelId(visibleState.params.checkpoint)" in CHAT_ADAPTER
    assert "resolveInferenceCheckpointId(status)" in CHAT_ADAPTER
    assert "skipAdoptServerModel: true" in CHAT_ADAPTER
    assert "snapshotQueuedChatRunSettings(" in CHAT_ADAPTER
    assert "...visibleExternalSettings" in CHAT_ADAPTER
    assert "visibleState.activeThreadEpoch" in CHAT_ADAPTER
    assert "activeThreadEpoch ===" in CHAT_ADAPTER
    assert "visibleState.queuedSettingsEpoch" in CHAT_ADAPTER
    assert "queuedSettingsEpoch ===" in CHAT_ADAPTER
    assert "preserveVisibleSettings: true" in CHAT_ADAPTER
    assert "captureResolvedRuntime: (runtime) =>" in CHAT_ADAPTER
    assert "applyAutoLoadRuntimeState(options" in CHAT_ADAPTER
    assert CHAT_ADAPTER.count("trackQueuedSettings: !options?.preserveVisibleSettings") >= 4
    assert "const visibleRoute = window.location.href" in CHAT_ADAPTER
    assert "window.location.href === visibleRoute" in CHAT_ADAPTER
    assert "{ trackQueuedSettings: false }" in CHAT_ADAPTER
    assert CHAT_ADAPTER.count("await resolveQueuedEmptyLocalModel(abortSignal)") >= 2
    assert "persist: !options?.preserveVisibleSettings" in CHAT_ADAPTER
    assert "beginModelLoading()" in CHAT_ADAPTER
    assert "endModelLoading(lifecycleLease)" in CHAT_ADAPTER
    lifecycle = _between(
        CHAT_ADAPTER,
        "async function resolveQueuedEmptyLocalModel(",
        "export function createOpenAIStreamAdapter",
    )
    assert lifecycle.index("beginModelLoading()") < lifecycle.index("await getInferenceStatus()")
    assert lifecycle.index("await getInferenceStatus()") < lifecycle.index(
        "await autoLoadSmallestModel("
    )
    assert "options?.abortSignal?.throwIfAborted()" in CHAT_ADAPTER
    assert CHAT_ADAPTER.count("await persistResolvedQueuedModel(params.checkpoint)") >= 2
    assert "notifyQueuedRunFailed();\n          throw error;" in CHAT_ADAPTER
    assert "pendingSettings.length === 1" not in QUEUED_SETTINGS
    assert "entry.threadIds.has(threadId)" in QUEUED_SETTINGS
    assert "return pendingSettings[index].settings" in QUEUED_SETTINGS
    assert "pendingSettings.splice(index, 1)[0].settings" not in QUEUED_SETTINGS
    assert "complete: discardOldestPendingSettings" in target
    assert "getActivePromptQueueItem(run)?.target.complete()" in THREAD
    assert "notifyPromptQueueRunFailed(resolvedThreadId ?? null)" in CHAT_ADAPTER
    assert CHAT_ADAPTER.index("const notifyQueuedRunFailed = () =>") < CHAT_ADAPTER.index(
        "if (\n        runtime.deepResearchEnabled"
    )
    image_gate = _between(
        CHAT_ADAPTER,
        "if (imageGateReason) {",
        "// Clear pending audio from store after extracting",
    )
    assert image_gate.index("notifyQueuedRunFailed();") < image_gate.index(
        "throw new Error(imageGateReason);"
    )
    assert "usesLocalModel:" in target
    assert "usePromptQueueUI.getState().byThreadId" in CONFIRM_MODEL_SWAP
    assert "getLocalPromptQueueThreadIds" in CONFIRM_MODEL_SWAP
    assert "promptQueueThreadIds" in MODEL_RUNTIME
    assert MODEL_RUNTIME.count("requestLocalPromptQueueStop(") >= 4
    assert MODEL_RUNTIME.index("requestLocalPromptQueueStop();") < MODEL_RUNTIME.index(
        "const loadResponse = await loadModel("
    )
    eject = _between(
        MODEL_RUNTIME,
        "const ejectModel = useCallback(",
        "return {",
    )
    assert "beginModelLoading()" in eject
    assert "endModelLoading(lifecycleLease)" in eject
    assert "beginModelLoading()" in SHARED_COMPOSER
    assert "endModelLoading(compareLifecycleLease)" in SHARED_COMPOSER
    assert SHARED_COMPOSER.count("releaseCompareModelLifecycle();") >= 3
    gpu_discovery = _between(
        SHARED_COMPOSER,
        "// Warm the device cache before the snapshot below",
        "// The GPU/offload knobs both compare loads must use",
    )
    assert "await ensureGpuDeviceCache();" in gpu_discovery
    assert "catch (error) {\n        releaseCompareModelLifecycle();" in gpu_discovery
    side_one = _between(
        SHARED_COMPOSER,
        "// Side 1: load → generate → wait",
        "// Side 2: load → generate → wait",
    )
    assert (
        side_one.index("const status1 = await ensureModelLoaded(model1)")
        < side_one.index("releaseCompareModelLifecycle();")
        < side_one.index("handle1.startRun()")
    )
    side_two = _between(
        SHARED_COMPOSER,
        "// Side 2: load → generate → wait",
        "compareStepSucceededRef.current = true",
    )
    assert (
        side_two.index("acquireCompareModelLifecycle();")
        < side_two.index("await confirmStopRunningChatsIfNeeded(")
        < side_two.index("compareStopDecision = currentStopDecision")
        < side_two.index("const status2 = await ensureModelLoaded(model2)")
        < side_two.index("releaseCompareModelLifecycle();")
        < side_two.index("handle2.startRun()")
    )
    assert "requestLocalPromptQueueStop" in eject
    assert "function promptQueueRunUsesLocalModel(run: PromptQueueRun)" in THREAD
    assert ".slice(Math.max(run.index, 0))" in THREAD
    assert ".some((item) => item.target.usesLocalModel)" in THREAD
    assert "local: promptQueueRunUsesLocalModel(run)" in THREAD
    assert "temporary: incognitoAtQueueStart" in THREAD
    assert "temporary: promptQueueRunIsTemporary(run)" in THREAD
    assert "dispatched: Boolean(getActivePromptQueueItem(run)?.dispatched)" in THREAD
    assert "queueEntry?.dispatched" in THREAD
    assert 'aria-label="Stop queued message"' in THREAD
    assert "entry.temporary" in QUEUE_BOUNDARY
    assert "localPromptQueueModelBoundary.advance()" in QUEUE_BOUNDARY
    assert "entry.local" in QUEUE_BOUNDARY
    assert "queuedRunSettings.params.checkpoint" in CHAT_ADAPTER
    assert "!runningByThreadId[threadId] && !cancel" in STOP_CHAT_THREAD
    assert "serverCancels.length === 0" in STOP_CHAT_THREAD
    assert "await confirmStopRunningChatsIfNeeded(" in SHARED_COMPOSER
    assert SHARED_COMPOSER.index("await confirmStopRunningChatsIfNeeded(") < SHARED_COMPOSER.index(
        'setText("");'
    )
    assert "requestLocalPromptQueueStop(" in SHARED_COMPOSER
    assert SHARED_COMPOSER.index("requestLocalPromptQueueStop(") < SHARED_COMPOSER.index(
        "const resp = await loadModel("
    )
    assert "force_cancel_active:" in SHARED_COMPOSER
    assert (
        "resolvedThreadId ===\n              useChatRuntimeStore.getState().activeThreadId"
        in CHAT_ADAPTER
    )
    assert (
        "findLatestUserAudioBase64(\n        survivingMessages,\n        !queuedRunSettings"
        in CHAT_ADAPTER
    )
    assert "if (audioBase64 && !queuedRunSettings)" in CHAT_ADAPTER
    assert ".setThreadContextUsage(usageThreadKey, usage)" in CHAT_ADAPTER
    assert (
        "usageThreadIsVisible &&\n"
        "            useChatRuntimeStore.getState().params.checkpoint === params.checkpoint"
        in CHAT_ADAPTER
    )


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
    assert "serverCancelByThreadId" in CLEAR_ALL_CHATS
    assert "stopChatThread(threadId)" in CLEAR_ALL_CHATS
    assert "detail: { threadIds, temporaryOnly: true }" in QUEUE_BOUNDARY
    assert "if (temporaryOnly)" in THREAD
    assert "threadIds !== undefined && threadIds.length === 0" in QUEUE_BOUNDARY
    assert "detail: threadIds ? { threadIds } : undefined" in QUEUE_BOUNDARY
    assert "const aliasesByQueuedRun = new Map<string, string[]>()" in CONFIRM_MODEL_SWAP
    assert "aliases.some((threadId) => runningIds.has(threadId))" in CONFIRM_MODEL_SWAP


def test_sidebar_exposes_queue_activity_for_each_thread():
    assert "const queueByThreadId = usePromptQueueUI((s) => s.byThreadId);" in APP_SIDEBAR
    assert "hasQueuedActivity" in APP_SIDEBAR
    assert "showWorkSpinner" in APP_SIDEBAR
    assert "{showWorkSpinner && (" in APP_SIDEBAR
    assert "hasUnreadActivity" in APP_SIDEBAR
    assert "clearChatNotifications(item)" in APP_SIDEBAR
