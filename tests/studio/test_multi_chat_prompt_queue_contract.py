# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for independent per-chat prompt queues."""

import re
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
QUEUED_MODEL_CAPABILITIES = (
    FRONTEND / "features/chat/utils/queued-model-capabilities.ts"
).read_text(encoding = "utf-8")
PRE_STREAM_RESERVATION = (FRONTEND / "features/chat/utils/pre-stream-run-reservation.ts").read_text(
    encoding = "utf-8"
)
CHAT_CLEAR_BOUNDARY = (FRONTEND / "features/chat/utils/chat-history-clear-boundary.ts").read_text(
    encoding = "utf-8"
)
CHAT_HISTORY_STORAGE = (FRONTEND / "features/chat/utils/chat-history-storage.ts").read_text(
    encoding = "utf-8"
)
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


def _guard_for(source: str, call: str) -> str:
    """The `if (...)` condition whose body performs ``call``.

    Searching the whole file for the identity comparison is not enough: the abort
    and cleanup branches next to the dispatch carry the same expression, so the
    dispatch guard could regress to `.has(reservationKey)` on its own and a
    whole-file search would still find a match in its neighbours.
    """
    assert call in source, f"missing call: {call}"
    head = source.split(call, 1)[0]
    opener = head.rfind("if (")
    assert opener != -1, f"no `if (` guarding {call}"
    return head[opener:]


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
    # The submit path decides whether to queue; the queueing itself lives in
    # queueComposerText, which #8952 extracted so the Cmd/Ctrl+Enter path could
    # share it. Assert the delegation here and the queueing there, rather than
    # expecting the call inline, so this stays a contract on behaviour instead
    # of on where the code happens to sit.
    assert "queueComposerText(liveThreadIsRunning || livePreStreamRunActive)" in submit

    queue_composer_text = _between(
        THREAD,
        "const queueComposerText = useCallback(",
        "const dismissWaitToast = useCallback(",
    )
    assert "startHydratedPromptQueue(" in queue_composer_text
    # Read into a local first: the send guard arms on the untrimmed value too,
    # since that is what a late DOM write carries.
    assert "const cleared = aui.composer().getState().text" in queue_composer_text
    assert "cleared.trim() !== queuedPrompt" in queue_composer_text
    assert "promptQueueStartPendingRef.current" in THREAD
    # Identity, not mere presence: a reservation can be replaced between the
    # start and the callback, and acting on the successor would dispatch the
    # wrong prompt. `.has` only asked whether the key was occupied.
    # Read out of the guard that actually dispatches, not out of the file: the
    # abort and cleanup branches beside it hold the same comparison, so a
    # whole-file search stays green while the dispatch alone regresses to `.has`.
    dispatch_guard = _guard_for(THREAD, "startPromptQueue(items, target, waitForCurrentRun);")
    assert "promptQueueStartPendingRef.current.get(reservationKey) ===" in dispatch_guard
    assert "promptQueueStartPendingRef.current.has(" not in dispatch_guard
    # The other two are load-bearing as well. Abort without the identity check
    # reports the successor's start as this one's failure; cleanup without it
    # deletes the successor's entry.
    assert THREAD.count("promptQueueStartPendingRef.current.get(reservationKey) ===") == 3
    assert "promptQueueStartPendingRef.current.delete(reservationKey)" in THREAD
    assert "promptQueueStartPendingRef.current.set(reservationKey, reservation)" in THREAD
    # Captured when the queue starts, not read live when it dispatches: a chat
    # toggled out of temporary mid-queue must not have its queued prompts
    # persisted, and reading the store at dispatch time would do exactly that.
    assert "const incognitoAtQueueStart = chatStateAtQueueStart.incognito" in THREAD
    assert "temporary: incognitoAtQueueStart" in THREAD
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
    assert "reservePreStreamRun(preStreamThreadIds, {" in THREAD
    assert "usesLocalModel:" in THREAD
    assert "aui.threads().__internal_getAssistantRuntime?.()" in THREAD
    assert "threads.getById(reservedThreadId).cancelRun()" in THREAD
    assert "adoptPreStreamRunReservation(token, preStreamThreadIds)" in THREAD
    assert "hasPreStreamRunReservation(getQueueThreadIds())" in THREAD
    append_failure = _between(
        THREAD,
        "function handleQueuedPromptAppendFailure(",
        "function consumePromptQueueDeepResearch(",
    )
    terminal_failure = append_failure.split(
        "if (item.dispatchRetries > PROMPT_QUEUE_MAX_DISPATCH_RETRIES)", 1
    )[1]
    assert terminal_failure.index("item.target.cancel();") < terminal_failure.index(
        "deletePromptQueueRun(run);"
    )
    assert terminal_failure.index("deletePromptQueueRun(run);") < terminal_failure.index(
        "item.target.complete();"
    )
    assert "releaseCurrentPreStreamRun();" in CHAT_ADAPTER
    assert "releasePreStreamRunReservation(reservationToken)" in CHAT_ADAPTER
    assert "class PreStreamAwareAttachmentAdapter" in RUNTIME_PROVIDER
    assert "preStreamRunThreadIdsForRuntime(" in RUNTIME_PROVIDER
    attachment_adapter = _between(
        RUNTIME_PROVIDER,
        "const attachments = useMemo(",
        "const adapters = useMemo(",
    )
    assert "[state.remoteId, state.id]" in attachment_adapter
    assert "useChatRuntimeStore.getState().activeThreadId" in attachment_adapter
    assert "preStreamRunThreadIdsForAdapter(" in CHAT_ADAPTER
    adapter_wrapper = CHAT_ADAPTER.rsplit("async *run(args)", 1)[1]
    assert "args.unstable_threadId," in adapter_wrapper
    assert "useChatRuntimeStore.getState().activeThreadId" in adapter_wrapper

    persisted_wrapper = _between(
        RUNTIME_PROVIDER,
        "function createPersistedRunAdapter(",
        "function useStudioRuntimeAdapters(",
    )
    assert persisted_wrapper.index(
        "const trackedRunStartThreadIds = runStartThreadIdsForMessages("
    ) < persisted_wrapper.index("findPreStreamRunReservation(reservationThreadIds)")
    assert "[options.unstable_threadId, ...trackedRunStartThreadIds]" in persisted_wrapper
    assert "findPreStreamRunReservation(reservationThreadIds)" in persisted_wrapper
    assert "await waitForRunStartHistoryAppend(options.messages)" in persisted_wrapper
    assert "releasePreStreamRunReservation(reservationToken)" in persisted_wrapper
    assert "notifyPromptQueueRunFailed(" in persisted_wrapper
    persisted_failure = _between(
        persisted_wrapper,
        "} catch (error) {",
        "throw error;",
    )
    assert persisted_failure.index("releasePreStreamRunReservation(reservationToken)") < (
        persisted_failure.index("notifyPromptQueueRunFailed(")
    )
    assert re.search(
        r"releasePreStreamRunReservation\(reservationToken\);\s*}\s*"
        r"//.*?notifyPromptQueueRunFailed\(",
        persisted_failure,
        re.S,
    ), "queue failure notification must not depend on a direct-send reservation"


def test_a_send_parked_on_the_settings_gate_queues_if_a_run_started_meanwhile():
    """The park is not the bug; releasing it into a running thread is.

    A submit that lands while a new chat's settings are pairing is parked with
    a "Loading this chat's settings" toast. When the gate closes, the release
    used to call `sendReservedComposer()` for anything that had not asked for
    the queue with Cmd/Ctrl+Enter -- even when a run had started in the
    meantime. The runtime refuses a send on a running thread, so the message
    was neither queued nor sent, and the wait toast had already been dismissed
    a few lines above: nothing on screen said the prompt was gone.

    Measured, not reasoned about. With the browser under an 8x CDP CPU
    throttle, so a build box renders like the 4 vCPU machines this shows up
    on, the app's own trace reads:

        +786 ms  submit -> settingsPending          (parked)
        +10480   release  text="..." running=true   (gate closed 236 ms later)
        +10482   release:sendReservedComposer

    and 90 seconds later: one user bubble, one /v1/chat/completions request,
    the prompt still sitting in the composer, no queue chip, no toast.

    The `forceQueue` branch already re-read `isRunning` for exactly this
    reason, in a comment that describes the bug in the branch beside it. The
    rule below is that the run check governs BOTH.
    """
    release = _between(
        THREAD,
        "// Fire the parked send once indexing clears",
        "// Drop any queued send + toast on unmount",
    )
    code = re.sub(r"//[^\n]*", "", release)
    assert "const waitForCurrentRun =" in code
    assert "aui.thread().getState().isRunning" in code, (
        "the release no longer asks whether a run started while the send was "
        "parked, so a parked prompt is sent into a streaming thread again"
    )
    # A pre-stream reservation is a run that has been accepted and has not
    # reached isRunning yet. handleSubmit treats it as running; so must this,
    # or the same prompt is lost in a narrower window.
    assert "hasPreStreamRunReservation(preStreamThreadIds)" in code

    # The gate on the queue branches, which is the fix itself: an active run
    # governs the release, not the Cmd/Ctrl+Enter intent.
    running = code.index("if (waitForCurrentRun) {")
    branch = code[running : code.index("if (forceQueue && !disableQueue) {")]
    assert "queueComposerText(true);" in branch, (
        "the release no longer queues behind the run that started while the "
        "send was parked, so the prompt goes back to being dropped silently"
    )
    # Every refusal handleSubmit makes, made here too. A parked send is the
    # same submit arriving late, so a branch it does not mirror is a state the
    # UI forbids being reachable through the settings gate.
    for rule, why in (
        (
            "if (disableQueue) {",
            "the project new-chat composer can queue again, binding the "
            "follow-up to a thread that does not exist yet",
        ),
        (
            "Only text prompts can be queued",
            "a parked send carrying an attachment falls through to a direct "
            "send while a run is live, which is the collision this branch "
            "exists to avoid",
        ),
    ):
        assert rule in branch, f"{why} (missing: {rule!r})"
    assert "sendReservedComposer" not in branch, (
        "the running branch still reaches a direct send; nothing that cannot "
        "be queued may be dispatched into a streaming thread"
    )

    # Research disables input outright -- handleSubmit returns before anything
    # else and the UI shows Stop research instead of Send.
    research = code.index("if (isResearchActive) {")
    assert research < running, (
        "the research refusal is not ahead of the queue path, so a prompt "
        "parked before research began starts a turn while it is still active"
    )
    assert "isResearchActive," in code, "isResearchActive is missing from the deps"
    assert "disableQueue," in code, "disableQueue is missing from the effect deps"

    # The draft outlives every path that does not complete. queueComposerText
    # clears it from its own onStarted callback, so clearing it up front loses
    # the text whenever the queue does not start -- a null target, an
    # invalidated start -- and after the composer is replaced it is gone.
    assert code.index("clearStoredDraft();") > code.index("if (forceQueue && !disableQueue) {"), (
        "the stored draft is cleared before the queue and refusal paths, so a "
        "prompt that is neither queued nor sent cannot be recovered"
    )
    # Unchanged: with nothing running the chord still queues, and an ordinary
    # send still sends. A fix that stopped sending would strand that case.
    assert "sendReservedComposer();" in code


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
    assert "!promptQueueTargetMountedRef.current" in target
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
    assert target.count("cancelled ||") >= 2
    assert target.count("!pendingSettingsIds.has(settingsId)") >= 2
    assert target.index("!pendingSettingsIds.has(settingsId)") < target.index(
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
        "const startDeepResearch = async function*",
        "const deepResearchHandoff",
    )
    assert "const sendTimeRuntime = runtime" in research
    assert "const liveRuntime = useChatRuntimeStore.getState()" not in research
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
    assert "liveRuntime.loadedContextLength" in auto_load_merge
    assert "isExternalModelId(visibleState.params.checkpoint)" in CHAT_ADAPTER
    assert "resolveInferenceCheckpointId(status)" in CHAT_ADAPTER
    assert "skipAdoptServerModel: true" in CHAT_ADAPTER
    assert "snapshotVisibleModelState(" in CHAT_ADAPTER
    assert "restoreVisibleModelState(visibleExternalState)" in CHAT_ADAPTER
    assert '"loadedContextLength"' in CHAT_ADAPTER
    assert '"maxContextLength"' in CHAT_ADAPTER
    assert '"nativeContextLength"' in CHAT_ADAPTER
    assert '"loadedIsMultimodal"' in CHAT_ADAPTER
    assert '"loadedIsDiffusion"' in CHAT_ADAPTER
    assert (
        '"contextUsage"'
        not in CHAT_ADAPTER[
            CHAT_ADAPTER.index("const VISIBLE_MODEL_RUNTIME_KEYS") : CHAT_ADAPTER.index(
                "] as const satisfies", CHAT_ADAPTER.index("const VISIBLE_MODEL_RUNTIME_KEYS")
            )
        ]
    )
    assert "contextUsage: liveUsage.contextUsage" in CHAT_ADAPTER
    assert "contextUsageByThreadId: liveUsage.contextUsageByThreadId" in CHAT_ADAPTER
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
    assert "trackQueuedSettings: false" in CHAT_ADAPTER
    assert "await resolveQueuedEmptyLocalModel(transitionSignal)" in CHAT_ADAPTER
    assert "await resolveQueuedEmptyLocalModel(abortSignal)" in CHAT_ADAPTER
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
    assert "getInferenceStatus().catch(() => null)" not in lifecycle
    assert "const status = await getInferenceStatus();" in lifecycle
    assert "options?.abortSignal?.throwIfAborted()" in CHAT_ADAPTER
    assert (
        len(
            re.findall(
                r"await persistResolvedQueuedModel\(\s*params\.checkpoint,"
                r"\s*runtime\.activeGgufVariant,\s*\)",
                CHAT_ADAPTER,
            )
        )
        >= 2
    )
    assert "notifyQueuedRunFailed" not in CHAT_ADAPTER
    assert "pendingSettings.length === 1" not in QUEUED_SETTINGS
    assert "entry.threadIds.has(threadId)" in QUEUED_SETTINGS
    assert "return pendingSettings[index].settings" in QUEUED_SETTINGS
    assert "pendingSettings.splice(index, 1)[0].settings" not in QUEUED_SETTINGS
    assert "complete: discardOldestPendingSettings" in target
    assert "getActivePromptQueueItem(run)?.target.complete()" in THREAD
    assert "adapterRunStartedSignals" not in CHAT_ADAPTER
    assert "pendingSettings.some((entry) => entry.threadIds.has(threadId))" in QUEUED_SETTINGS
    queued_run_failure = _between(
        CHAT_ADAPTER,
        "try {\n        yield* adapter.run(args);",
        "} finally {",
    )
    assert "if (!args.abortSignal.aborted)" in queued_run_failure
    assert queued_run_failure.index("notifyPromptQueueRunFailed(") < queued_run_failure.index(
        "throw error;"
    )
    queue_failure_handler = _between(
        THREAD,
        "function handlePromptQueueRunFailed(",
        'if (typeof window !== "undefined")',
    )
    assert "if (failedRun)" in queue_failure_handler
    assert "retainPendingPromptQueueItemsAfterFailure(failedRun)" in queue_failure_handler
    assert "deletePromptQueueRun(failedRun);" in queue_failure_handler
    retained_failure = _between(
        THREAD,
        "function retainPendingPromptQueueItemsAfterFailure(run: PromptQueueRun)",
        "function cancelPendingPromptQueueFactoriesForStop<",
    )
    assert retained_failure.index("activeItem.target.complete();") < retained_failure.index(
        "run.items.splice(activeIndex, 1);"
    )
    assert "waitForPromptQueueTargetIdle(run);" in retained_failure
    local_queue_stop = _between(
        THREAD,
        "function stopLocalPromptQueueRun(run: PromptQueueRun)",
        "function stopLocalPromptQueueRunsForThreadIds(threadIds: string[])",
    )
    assert "if (plan.refreshTargetIdleWait)" in local_queue_stop
    assert "refreshPromptQueueTargetIdleWait(run);" in local_queue_stop
    assert "claimPreStreamRunReservation(reservationToken);" in RUNTIME_PROVIDER
    assert "if (!reservation.claimed)" in PRE_STREAM_RESERVATION
    assert "loadedIsMultimodal: isMultimodalResponse(status)" in lifecycle
    assert "isAudio: status.is_audio ?? false" in lifecycle
    assert "hasAudioInput: status.has_audio_input ?? false" in lifecycle
    assert CHAT_ADAPTER.count("models: mergeQueuedModelCapabilities(") == 2
    assert "modelIndex === index ? { ...model, ...capabilities } : model" in (
        QUEUED_MODEL_CAPABILITIES
    )
    assert "loadedIsMultimodal: state.loadedIsMultimodal" in CHAT_ADAPTER
    assert "queuedEmptyModelRuntime?.loadedIsMultimodal" in auto_load_merge
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
    select_model = _between(
        MODEL_RUNTIME,
        "const selectModel = useCallback(",
        "const ejectModel = useCallback(",
    )
    assert (
        select_model.index("beginModelLoading()")
        < select_model.index("await confirmStopRunningChatsIfNeeded(")
        < select_model.index("cancelPreStreamRunReservations(stopDecision.preStreamRunTokens)")
        < select_model.index("requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds)")
    )
    assert "beginModelLoading()" in eject
    assert "endModelLoading(lifecycleLease)" in eject
    assert "beginModelLoading()" in SHARED_COMPOSER
    assert "endModelLoading(compareLifecycleLease)" in SHARED_COMPOSER
    assert SHARED_COMPOSER.count("releaseCompareModelLifecycle();") >= 3
    compare_upgrade = _between(
        SHARED_COMPOSER,
        "const upgraded = await confirmTransformersUpgradeIfNeeded({",
        "});",
    )
    assert "forceCancelActive:" in compare_upgrade
    assert "compareStopDecision?.forceCancelActive ?? false" in compare_upgrade
    compare_handle = _between(
        SHARED_COMPOSER,
        "export function RegisterCompareHandle(",
        "type PendingImage =",
    )
    assert "aui.threads().__internal_getAssistantRuntime?.()" in compare_handle
    assert "runtime?.threads.getById(threadId)" in compare_handle
    assert "thread.subscribe(" in compare_handle
    assert "useChatRuntimeStore.subscribe(" not in compare_handle
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
    assert (
        eject.index("beginModelLoading()")
        < eject.index("await confirmStopRunningChatsIfNeeded(")
        < eject.index("cancelPreStreamRunReservations(stopDecision.preStreamRunTokens)")
        < eject.index("requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds)")
    )
    assert "function promptQueueRunUsesLocalModel(run: PromptQueueRun)" in THREAD
    assert ".slice(Math.max(run.index, 0))" in THREAD
    assert ".some((item) => item.target.usesLocalModel)" in THREAD
    assert "local: promptQueueRunUsesLocalModel(run)" in THREAD
    assert "detail: { threadIds, localOnly: true }" in QUEUE_BOUNDARY
    assert "stopLocalPromptQueueRunsForThreadIds(threadIds ?? [])" in THREAD
    local_queue_stop = _between(
        THREAD,
        "function stopLocalPromptQueueRun(run: PromptQueueRun)",
        "function stopLocalPromptQueueRunsForThreadIds(threadIds: string[])",
    )
    assert "planLocalPromptQueueStop(" in local_queue_stop
    assert "activeItem?.target.cancel();" in local_queue_stop
    assert "waitForPromptQueueTargetIdle(run);" in local_queue_stop
    pending_factory_stop = _between(
        THREAD,
        "function cancelPendingPromptQueueFactoriesForStop<",
        "function stopAllPromptQueueRuns()",
    )
    assert pending_factory_stop.index("if (localOnly)") < pending_factory_stop.index(
        "for (const [key, reservation]"
    )
    assert "cancelPendingPromptQueueFactoriesForStop(" in THREAD
    assert "temporary: incognitoAtQueueStart" in THREAD
    assert "temporary: promptQueueRunIsTemporary(run)" in THREAD
    assert "dispatched: Boolean(getActivePromptQueueItem(run)?.dispatched)" in THREAD
    assert "queueEntry?.dispatched" in THREAD
    assert 'aria-label="Stop queued message"' in THREAD
    assert "entry.temporary" in QUEUE_BOUNDARY
    assert "localPromptQueueModelBoundary.advance()" in QUEUE_BOUNDARY
    assert "entry.local" in QUEUE_BOUNDARY
    assert "queuedRunSettings.params.checkpoint" in CHAT_ADAPTER
    persisted_adapter = _between(
        RUNTIME_PROVIDER,
        "function createPersistedRunAdapter(",
        "function useStudioRuntimeAdapters(",
    )
    assert "const trackedRunStartThreadIds = runStartThreadIdsForMessages(" in persisted_adapter
    assert "isPreStreamRunReservationCancelled(reservationToken)" in persisted_adapter
    assert persisted_adapter.count("throwIfReservationCancelled();") == 2
    assert persisted_adapter.index(
        "requestPromptQueueStop(persistedRunThreadIds)"
    ) < persisted_adapter.index("notifyPromptQueueRunFailed(")
    assert "pendingRunStartThreadIdsByMessageId" in RUNTIME_PROVIDER
    assert "localThreadId," in RUNTIME_PROVIDER
    successful_persisted_preflight = _between(
        RUNTIME_PROVIDER,
        "async function waitForRunStartHistoryAppend(",
        "function createPersistedRunAdapter(",
    )
    assert successful_persisted_preflight.index(
        "pendingRunStartReadyByMessageId.delete(userMessage.id)"
    ) < successful_persisted_preflight.index(
        "pendingRunStartThreadIdsByMessageId.delete(userMessage.id)"
    )
    assert "!runningByThreadId[threadId] && !cancel" in STOP_CHAT_THREAD
    assert "serverCancels.length === 0" in STOP_CHAT_THREAD
    assert "await confirmStopRunningChatsIfNeeded(" in SHARED_COMPOSER
    send_flow = _between(
        SHARED_COMPOSER,
        "async function send()",
        "sendRef.current = send;",
    )
    assert "const submittedText = text;" in send_flow
    assert "const submittedImages = pendingImages;" in send_flow
    assert "const submittedAudio = pendingAudio;" in send_flow
    assert "textRef.current === submittedText" in send_flow
    assert "pendingImagesRef.current === submittedImages" in send_flow
    assert "pendingAudioRef.current === submittedAudio" in send_flow
    confirm_index = send_flow.index("await confirmStopRunningChatsIfNeeded(")
    first_draft_check = send_flow.index("if (!submittedDraftIsCurrent())")
    gpu_discovery_index = send_flow.index("await ensureGpuDeviceCache();")
    second_draft_check = send_flow.index(
        "if (!submittedDraftIsCurrent())",
        gpu_discovery_index,
    )
    assert (
        send_flow.index("beginModelLoading()")
        < confirm_index
        < first_draft_check
        < gpu_discovery_index
        < second_draft_check
        < send_flow.index("clearSubmittedDraft();")
    )
    assert "requestLocalPromptQueueStop(" in SHARED_COMPOSER
    assert "compareStopDecision?.preStreamRunTokens ?? []" in SHARED_COMPOSER
    assert SHARED_COMPOSER.index("requestLocalPromptQueueStop(") < SHARED_COMPOSER.index(
        "const resp = await loadModel("
    )
    apply_compare_stop = _between(
        send_flow,
        "const applyCompareStopDecision = () => {",
        "// Helper: load a model and update store checkpoint",
    )
    assert "cancelPreStreamRunReservations(" in apply_compare_stop
    assert "compareStopDecision?.preStreamRunTokens ?? []" in apply_compare_stop
    assert "requestLocalPromptQueueStop(" in apply_compare_stop
    assert "compareStopDecision?.promptQueueThreadIds" in apply_compare_stop
    ensure_compare_model = _between(
        send_flow,
        "async function ensureModelLoaded(",
        "// Side 1: load",
    )
    already_active = _between(
        ensure_compare_model,
        "if (isAlreadyActive && !config && !loadedFromConfig) {",
        "}",
    )
    assert already_active.index("applyCompareStopDecision();") < already_active.index(
        'return "ready";'
    )
    assert ensure_compare_model.count("applyCompareStopDecision();") == 2
    validated_load_stop = ensure_compare_model.rindex("applyCompareStopDecision();")
    assert (
        ensure_compare_model.index("const validation = await validateModel(")
        < validated_load_stop
        < ensure_compare_model.index("const resp = await loadModel(")
    )
    assert "force_cancel_active:" in SHARED_COMPOSER
    assert (
        "useChatRuntimeStore.getState().activeThreadId ===\n"
        "          (usageThreadKey ?? activeThreadIdAtRunStart)" in CHAT_ADAPTER
    )
    assert (
        "findLatestUserAudioBase64(\n        survivingMessages,\n        !queuedRunSettings"
        in CHAT_ADAPTER
    )
    assert "if (audioBase64 && !queuedRunSettings)" in CHAT_ADAPTER
    assert ".setThreadContextUsage(usageThreadKey, usage)" in CHAT_ADAPTER
    assert re.search(
        r"usageThreadIsVisible\s*&&\s*"
        r"useChatRuntimeStore\.getState\(\)\.params\.checkpoint\s*===\s*params\.checkpoint",
        CHAT_ADAPTER,
    )


def test_base64_media_turns_stay_on_the_legacy_stream():
    candidate = _between(
        CHAT_ADAPTER,
        "const generationCandidate = Boolean(",
        ");",
    )
    assert "!imageBase64" in candidate
    assert "!audioBase64" in candidate
    assert "!videoBase64" in candidate


def test_continuations_stay_on_the_legacy_stream():
    """Continue yields its seeded partial before the request starts.

    That autosave can reach storage before durable admission does, and admission refuses a
    placeholder that already has content with a 409, which is not one of the errors that
    falls back to the legacy stream. So the turn would fail outright rather than generate.
    """
    candidate = _between(
        CHAT_ADAPTER,
        "const generationCandidate = Boolean(",
        ");",
    )
    assert "!continuation" in candidate


def test_compare_prompt_list_resets_when_preflight_never_starts_a_run():
    reset = _between(
        SHARED_COMPOSER,
        "function resetPromptQueue()",
        "function advanceQueue()",
    )
    assert "isQueueRunningRef.current = false;" in reset
    assert "setIsQueueRunning(false);" in reset
    assert "queueRef.current = [];" in reset
    assert "queueIndexRef.current = 0;" in reset
    assert "setQueueProgress({ current: 0, total: 0 });" in reset

    send_flow = _between(
        SHARED_COMPOSER,
        "async function send()",
        "sendRef.current = send;",
    )
    unavailable_lifecycle = _between(
        send_flow,
        "if (compareLifecycleLease === null)",
        "const releaseCompareModelLifecycle = () =>",
    )
    assert "resetPromptQueue();" in unavailable_lifecycle

    failed_preflight = _between(
        send_flow,
        "compareStopDecision = await confirmStopRunningChatsIfNeeded(",
        "if (!compareStopDecision.proceed)",
    )
    assert "resetPromptQueue();" in failed_preflight

    declined_preflight = _between(
        send_flow,
        "if (!compareStopDecision.proceed)",
        "if (!submittedDraftIsCurrent())",
    )
    assert (
        declined_preflight.index("releaseCompareModelLifecycle();")
        < declined_preflight.index("resetPromptQueue();")
        < declined_preflight.index("return;")
    )

    changed_draft = _between(
        send_flow,
        "const keepChangedDraft = () =>",
        "const clearSubmittedDraft = () =>",
    )
    assert "releaseCompareModelLifecycle();" in changed_draft
    assert "resetPromptQueue();" in changed_draft

    failed_gpu_discovery = _between(
        send_flow,
        "// Warm the device cache before the snapshot below",
        "// The GPU/offload knobs both compare loads must use",
    )
    assert "resetPromptQueue();" in failed_gpu_discovery

    compare_run = _between(
        send_flow,
        "setComparing(true);",
        "} else {",
    )
    failed_compare = _between(compare_run, "} catch (err) {", "} finally {")
    assert "compareStepSucceededRef.current = false;" in failed_compare
    assert "resetPromptQueue();" in failed_compare


def test_clear_all_invalidates_and_removes_late_fresh_thread_initialization():
    target = _between(
        THREAD,
        "const createPromptQueueTarget = useCallback(",
        "const dismissWaitToast",
    )
    assert "const historyClearGeneration = chatHistoryClearBoundary.capture()" in target
    assert "chatHistoryClearBoundary.capture() !== historyClearGeneration" in target
    assert "if (initializingFreshThread)" in target
    assert "initializedFreshThreadId = remoteId" in target
    assert "freshThreadAppendAccepted = true" in target
    assert "removeFreshThreadPersistedAfterAbort()" in target
    assert "removeFreshThreadPersistedAfterAbort(true)" not in target
    assert "markChatThreadDeleted(initializedFreshThreadId)" in target
    assert "deleteStoredChatThreads([initializedFreshThreadId])" in target
    assert "aui.threads().switchToNewThread()" in target
    assert "chatHistoryClearBoundary.advance();" in CLEAR_ALL_CHATS
    assert CLEAR_ALL_CHATS.index("chatHistoryClearBoundary.advance();") < CLEAR_ALL_CHATS.index(
        "requestPromptQueueStop();"
    )
    # Matched on the call prefix, not the whole call: #8932 gave clearStoredChats an options
    # argument, which changes nothing about the ordering this pins.
    assert CLEAR_ALL_CHATS.index("requestPromptQueueStop();") < CLEAR_ALL_CHATS.index(
        "return await clearStoredChats("
    )
    assert "const historyClearGeneration = chatHistoryClearBoundary.capture();" in RUNTIME_PROVIDER
    assert "await throwIfHistoryWasCleared(initialized.remoteId);" in RUNTIME_PROVIDER
    assert "await throwIfHistoryWasCleared(remoteId);" in RUNTIME_PROVIDER
    assert "trackStoredChatThreadRecord(" in RUNTIME_PROVIDER
    assert "class ChatHistoryClearBoundary" in CHAT_CLEAR_BOUNDARY
    assert "capture(): number" in CHAT_CLEAR_BOUNDARY
    assert "advance(): number" in CHAT_CLEAR_BOUNDARY
    assert "const reopenAdmission = threadRecordWrites.closeAdmission();" in CHAT_HISTORY_STORAGE
    assert (
        "const pendingThreadIds = threadRecordWrites.idsRequiringFence();" in CHAT_HISTORY_STORAGE
    )
    assert "tombstoneThreadIds: idsToFence" in CHAT_HISTORY_STORAGE
    assert "threadRecordWrites.confirmFinalState(idsToFence);" in CHAT_HISTORY_STORAGE


def test_a_failed_thread_row_write_surfaces_to_the_patch_caller():
    """A retry that reports undefined reads as "no row to update", so the queued run's
    model correction is dropped and never retried: thread.tsx clears
    shouldCorrectPersistedModel right after the awaited updateStoredChatThread."""
    retry = _between(
        CHAT_HISTORY_STORAGE,
        "async function retryFailedThreadRecord(",
        "export async function listStoredChatMessages(",
    )
    # awaiting the tracked write, not the settle-all helper, is what propagates the failure
    assert "await trackStoredChatThreadRecord(threadId, createRecord);" in retry
    assert "await awaitStoredChatThreadWrites(threadId);\n  return" not in retry


def test_noop_setting_refreshes_do_not_invalidate_pending_queues():
    assert "shouldAdvanceQueuedSettingsEpoch(" in CHAT_RUNTIME_STORE
    set_params = _between(CHAT_RUNTIME_STORE, "setParams: (params, options)", "setCustomPresets:")
    assert "state.params," in set_params
    assert "params," in set_params
    assert "queuedSettingsChanged" in set_params
    set_checkpoint = _between(
        CHAT_RUNTIME_STORE,
        "setCheckpoint: (modelId, ggufVariant, options)",
        "setActiveThreadId:",
    )
    assert "nextGgufVariant" in set_checkpoint
    assert "nextDeepResearchEnabled" in set_checkpoint
    assert "queuedSettingsChanged" in set_checkpoint


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


def test_a_backgrounded_pane_autosaves_without_naming_itself_active():
    """The shared provider (#9129) keeps a hidden pane's run alive, so its autosave still fires
    after Compare has hidden it. The SAVE must keep happening; only the active-thread PUBLICATION
    is suppressed, or the hidden base pane writes its own remote id into the store and Compare's
    ``exportThreadIds = [model1, model2, activeThreadId]`` downloads the unrelated base
    conversation alongside the two compare threads.
    """
    autosave = _between(
        RUNTIME_PROVIDER,
        "function ThreadBackendAutosave(",
        "\nexport function useChatActive(",
    )

    # The pane knows it is hidden.
    assert (
        "backgrounded: boolean;" in autosave
    ), "ThreadBackendAutosave has to be told, like every other sync component here"
    assert (
        "backgrounded={backgrounded}" in RUNTIME_PROVIDER
    ), "and the provider has to pass it, or the prop is inert"

    # Read at publish time, not captured when the save was queued: the save that publishes
    # may have been scheduled while the pane was on screen and resolve long after Compare
    # hid it.
    assert "const backgroundedRef = useRef(backgrounded);" in autosave
    assert "backgroundedRef.current = backgrounded;" in autosave

    # The publication is what is gated -- and ONLY the publication.
    assert (
        "!backgroundedRef.current &&" in autosave
    ), "the active-thread publication must be gated on the pane being visible"
    publish_at = autosave.index("store.setActiveThreadId(remoteId)")
    guard_at = autosave.index("!backgroundedRef.current")

    # ...and on no switch away from this thread being in flight. switchToNewThread() is
    # async, so mainThreadId still reads as this pane for the whole gap, and a save landing
    # in it republishes the chat the user just left into the view they navigated to.
    assert "!switchInFlight" in autosave, (
        "the publication must also stand down while this provider's own New Chat switch is "
        "still resolving"
    )
    assert (
        "switchState.landedAttempt !== switchState.attempt" in autosave
    ), "the in-flight window is attempt != landedAttempt, not merely activeNonce being set"
    assert guard_at < publish_at, "the guard has to come before the write it guards"

    # The save itself is untouched: gating it would defeat the PR, which exists so a run
    # that outlives its view still lands on disk.
    for call in (
        "await ensureStoredChatThread(remoteId)",
        "await syncExportedRepositoryToBackend(remoteId, exported)",
    ):
        assert call in autosave, f"{call} must still run while backgrounded"
        assert autosave.index(call) < guard_at, f"{call} must not sit behind the visibility guard"


def test_the_history_adapters_publish_stands_down_with_the_autosaves():
    """``ThreadBackendAutosave`` is not the only place a pane names itself the active thread.

    The history adapter's ``append()`` publishes the same id for every persisted message,
    including the assistant message of the background run #9129 exists to keep alive.
    ``enterCompare`` blanks the active id, so the ``!== remoteId`` test passes and a hidden pane
    republishes itself into the same ``exportThreadIds`` the autosave guard was added for. Both
    must stand down together, or gating one is decorative.
    """
    append = _between(
        RUNTIME_PROVIDER,
        "      append({ parentId, message }: ExportedMessageRepositoryItem) {",
        "\n  // Always register the adapter so the mic stays clickable",
    )

    assert (
        "store.setActiveThreadId(remoteId);" in append
    ), "this test is about the history adapter's publication; if it moved, follow it"
    assert (
        "!backgroundedRef?.current &&" in append
    ), "the history adapter's publication needs the same visibility gate as the autosave's"
    assert "!switchInFlight" in append, (
        "...and the same stand-down while a New Chat switch this provider started is still "
        "resolving; see the autosave test for why mainThreadId cannot be trusted in that gap"
    )

    # Read at publish time, through a ref, for the same reason the autosave does: the write
    # is queued when the message arrives and resolves after Compare may have hidden the pane.
    assert "const backgroundedRef = useRef(backgrounded);" in RUNTIME_PROVIDER
    assert "backgroundedRef.current = backgrounded;" in RUNTIME_PROVIDER

    # ...and the ref has to actually reach it. A ref rather than the boolean, so handing it
    # down cannot change the memoized runtime hook's identity and rebuild the runtime.
    hook_build = _between(
        RUNTIME_PROVIDER,
        "  const runtimeHook = useMemo(",
        "  const runtime = useRemoteThreadListRuntime({",
    )
    assert "backgroundedRef," in hook_build, "createRuntimeHook has to be handed the ref"
    assert "[initialThreadId, modelType, onInitialHistoryReady, pairId]," in hook_build, (
        "and the ref must NOT join the dependency array: a new hook identity rebuilds the "
        "runtime, which is the one thing the shared provider must never do"
    )

    # The write itself is untouched. Only the publication is gated.
    assert "await awaitStoredChatThreadWrites(remoteId);" in append
    assert append.index("await awaitStoredChatThreadWrites(remoteId);") < append.index(
        "!backgroundedRef?.current"
    ), "persisting a background run's message must not sit behind the visibility guard"
