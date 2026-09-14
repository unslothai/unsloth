# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend" / "src"


def source(path: str) -> str:
    return (FRONTEND / path).read_text(encoding = "utf-8")


def test_research_api_is_isolated_and_cursor_based() -> None:
    api = source("features/chat/api/research-api.ts")
    store = source("features/chat/stores/research-run-store.ts")
    assert 'authFetch("/api/chat/research-runs"' in api
    assert "authFetch(`/api/chat/research-runs/active?${query}`)" in api
    assert "const { runs, hasRun }" in api
    assert "runs.at(-1) ?? null" in api
    assert "getResearchThreadState" in api
    assert "/events?after=${Math.max(0, after)}" in api
    # POST, not GET: proxies that buffer a streamed GET leave the activity panel empty.
    assert 'method: "POST", headers: { accept: "text/event-stream" }' in api
    assert "export async function* followResearchRun" in api
    assert "Math.min(8_000, 500 * 2 ** (failures - 1))" in api
    assert "for await (const event of streamResearchEvents" in api
    assert 'source: "event"' in api
    assert "fresh.report !== currentRun.report" in api
    assert "await waitForReconnect(" in api
    assert "while (!(run || signal?.aborted))" in api
    assert "isPermanentResearchError(error)" in api
    assert 'yield { run, source: "snapshot" }' in api
    assert "event.id <= pending.event.id" in store
    # The store owns the stream, so nothing else can stall ingestion by not reading.
    assert "ensureResearchRunFollowed(run.id, run);" in store
    assert "export async function* watchResearchRun" in store
    for action in ("cancel", "retry"):
        assert f'mutate(id, "{action}")' in api
    assert 'mutate(id, "approve", { planRevision, planHash })' in api
    assert "JSON.stringify({ plan, expectedRevision })" in api


def test_research_mode_is_single_chat_and_detaches_without_cancel() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")

    inference_request = source("features/chat/research-inference-request.ts")
    thread = source("components/assistant-ui/thread.tsx")
    assert "runtime.deepResearchEnabled" in adapter
    assert "!options.pairId" in adapter
    assert 'options.modelType === "base"' in adapter
    assert "cancelResearchRun(run.id)" not in adapter
    assert "createResearchRun" in adapter
    assert "await saveStoredChatMessage({" in adapter
    assert "unstable_assistantMessageId," in adapter
    assert "if (!unstable_assistantMessageId)" in adapter
    assert "assistantMessageId: unstable_assistantMessageId" in adapter
    # the adapter reads store state and never owns the stream, so a stalled reader cannot freeze it.
    assert "watchResearchRun(createdRun.id" in adapter
    assert "followResearchRun" not in adapter
    assert "inferenceRequest" in adapter
    assert "Number.isFinite(input.temperature)" in inference_request
    assert "Number.isFinite(input.topP)" in inference_request
    assert "Number.isFinite(input.maxTokens)" in inference_request
    assert "Math.min(8192, Math.floor(input.maxTokens))" in inference_request
    assert '{ type: "text" as const, text: report }' in adapter
    # yields are deduped by status, or every streamed delta drives an autosave the server rejects.
    assert "run.status === yieldedStatus" in adapter
    # runSignal, not abortSignal: each run gets its own controller, forwarded from the thread
    # signal, so one chat's Stop cannot abort a sibling streaming in the background.
    assert "if (runSignal.aborted) return" in adapter
    # The model decides: arming research offers it the deep_research tool, and the run starts
    # off the tool events every loop already publishes, never off the toggle alone.
    assert "deep_research_armed: true" in adapter
    assert 'toolEvent.tool_name === "deep_research"' in adapter
    assert "readDeepResearchToolEvent(deepResearchHandoff, toolEvent)" in adapter
    assert "if (deepResearchHandoff.question !== null && !runSignal.aborted)" in adapter
    assert 'yield* startDeepResearch("")' not in adapter
    # Armed research asks for Studio's tool loop on the external body too. Without it the
    # turn proxies through, the model is never offered the tool, and arming does nothing.
    assert "projectRagEnabled ||" in adapter
    assert "deepResearchArmed)" in adapter
    research = adapter.split("const startDeepResearch = async function*", 1)[1].split(
        "const sandboxSessionId", 1
    )[0]
    assert "await resolveQueuedEmptyLocalModel(transitionSignal)" in research
    assert "queuedEmptyModelRuntime = resolution.modelRuntime" in research
    assert "signal: researchFollowController.signal" in adapter
    assert "beginExternalResearchFollow(" in adapter
    assert "ragScope" in adapter
    assert "const projectRagEnabled = researchProjectId" in adapter
    assert "runtime.ragEnabled || projectRagEnabled" in adapter
    submit = thread.split("const handleSubmit = useCallback", 1)[1].split("const stopQueue", 1)[0]
    assert "if (isResearchActive)" in submit
    assert "event.preventDefault()" in submit
    assert "runtime.ragEnabled\n                    ? { thread_id: resolvedThreadId }" in adapter
    message_error = thread.split("const MessageError: FC = () =>", 1)[1].split(
        "const GeneratingIndicator", 1
    )[0]
    assert "useThreadResearchActive()" in message_error
    assert "!researchRunId && !researchActive" in message_error
    create_block = adapter.split("createdRun = await createResearchRun({", 1)[1].split("});", 1)[0]
    assert "modelId:" not in create_block
    assert "prompt," not in create_block
    assert "instructions: researchInstructions" in create_block
    assert "resolveChatInstructions" in adapter


def test_research_handoff_transition_honors_the_original_run_stop() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    research = adapter.split("const startDeepResearch = async function*", 1)[1].split(
        "const deepResearchHandoff = newDeepResearchHandoff();", 1
    )[0]

    assert "transitionSignal: AbortSignal = abortSignal" in research
    assert "await waitForModelReady(transitionSignal)" in research
    assert "await resolveQueuedEmptyLocalModel(transitionSignal)" in research
    registered = research.index(
        "runtime.registerThreadServerCancel(threadKey, researchServerCancel);"
    )
    assert research.rfind("if (transitionSignal.aborted) return;", 0, registered) >= 0
    assert adapter.count("startDeepResearch(deepResearchHandoff.question, runSignal)") == 2


def test_research_reasoning_effort_is_clamped_to_the_loaded_model() -> None:
    # A level the loaded model lacks is dropped by llama.cpp, so the durable run would silently fall back to the
    # template default. Must use the same helper and levels as normal local chat so the two paths cannot drift
    # apart again.
    adapter = source("features/chat/api/chat-adapter.ts")
    inference_request = source("features/chat/research-inference-request.ts")
    assert "buildResearchInferenceRequest({" in adapter
    assert "request.reasoningEffort = input.reasoningEffort;" not in inference_request
    assert "request.reasoningEffort = input.clampReasoningEffort(" in inference_request
    assert "input.reasoningEffortLevels," in inference_request
    assert "const localReasoningEffort = clampReasoningEffortToLevels(" in adapter


def test_research_presave_keeps_the_follow_up_parent() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    presave = adapter.split("const userMessage =", 1)[1].split(
        "const createdRun = await createResearchRun({", 1
    )[0]

    assert "const userMessageIndex = messages.indexOf(userMessage);" in presave
    assert "const userMessageParentId =" in presave
    assert "userMessageIndex > 0 ? messages[userMessageIndex - 1]!.id : null" in presave
    # A stored null is an edited root; `??` would reparent it under the predecessor.
    assert "storedUserMessage && storedUserMessage.parentId !== undefined" in presave
    assert "? storedUserMessage.parentId" in presave
    assert ": userMessageParentId," in presave
    assert "parentId: storedUserMessage?.parentId ?? userMessageParentId" not in presave
    assert "parentId: storedUserMessage?.parentId ?? null" not in presave


def test_research_metadata_and_server_merge_are_persisted() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    runtime = source("features/chat/runtime-provider.tsx")
    assert "researchRunId: run.id" in adapter
    assert "serverManaged: true" in adapter
    assert re.search(r"getResearchThreadState\(\s*remoteId,?\s*\)", runtime)
    assert "preserveServerManaged" in runtime
    assert "sameResearchRun" in runtime
    assert "existingRevision > incomingRevision" in runtime
    assert "const userMessage = [...messages]" in runtime
    assert '.find((message) => message.role === "user")' in runtime
    assert "pendingRunStartReadyByMessageId.get(userMessage.id)" in runtime


def test_research_presentation_is_integrated() -> None:
    thread = source("components/assistant-ui/thread.tsx")
    page = source("features/chat/chat-page.tsx")
    chat_index = source("features/chat/index.ts")
    store = source("features/chat/stores/chat-runtime-store.ts")
    activity = source("features/chat/components/research-activity-panel.tsx")
    message = source("features/chat/components/research-message.tsx")
    markdown_preview = source("components/markdown/markdown-preview.tsx")
    safe_markdown_url = source("lib/safe-markdown-url.ts")
    coordinator = source("features/chat/stores/research-run-store.ts")
    assert "DeepResearchComposerButton" in thread
    assert "Deep research" in thread
    research_gate = thread.split("const researchDisabled =", 1)[1].split(";", 1)[0]
    assert "!modelLoaded" not in research_gate
    assert "<ResearchMessage />" in thread
    assert "if (researchRunId) return null" in thread
    assert "!researchRunId &&" in thread
    assert "if (researchRunId || ownsResearchMessage)" in thread
    assert "ResearchMessageRunIdContext = createContext<string | null>(null)" in thread
    assert "researchReplyOwnsRun(boundResearchAssistantMessageId, messageId)" in thread
    assert "<ResearchMessageRunIdContext.Provider value={researchRunId}>" in thread
    assert "useContext(ResearchMessageRunIdContext)" in thread
    # A prompt whose reply is a research message loses its edit and delete controls.
    owners = source("components/assistant-ui/research-reply-owners.ts")
    assert "researchReplyOwners(" in thread
    assert "() => aui.thread().export().messages" in thread
    # An empty run id still means "no research reply", as Boolean() rather than a null check.
    assert "Boolean(getResearchRunId(metadata))" in thread
    assert "isResearchReply(message.metadata)" in owners
    assert "owners.add(parentId)" in owners
    user_actions = thread.split("const UserActionBar: FC = () =>", 1)[1].split(
        "const EditComposer:", 1
    )[0]
    assert "!ownsResearchMessage &&" in user_actions
    assert "<ActionBarPrimitive.Edit" in user_actions
    message_error = thread.split("const MessageError: FC = () =>", 1)[1].split(
        "const GeneratingIndicator:", 1
    )[0]
    assert "!researchRunId &&" in message_error
    assert "ResearchActivityPanel" in page
    assert "ResearchActivitySheet" in page
    assert "ResearchActivityPanel" in chat_index
    assert 'role="log"' in activity
    assert "Review the research plan" in activity
    assert "Start research" in activity
    assert "cancelResearchRun" in thread
    assert "Stop research" not in activity
    assert "retryResearchRun" in activity
    assert "Deep research completed" in message
    empty_fallback = message.split("if (!run) {", 1)[1].split("if (run.status", 1)[0]
    assert (
        empty_fallback.index("if (fallbackText.trim())")
        < empty_fallback.index("if (!ownsRun)")
        < empty_fallback.index("Loading research…")
    )
    assert "return null;" in empty_fallback
    assert "<DocumentSourcesGroup" in message
    assert "urlTransform={safeMarkdownUrl}" in markdown_preview
    assert 'node.tagName !== "img"' in safe_markdown_url
    assert "ensureResearchRunFollowed" in coordinator
    assert "reasoning.updated" in coordinator
    assert "source.added" in coordinator
    assert 'activity.state === "running"' in coordinator
    assert "terminalState" in coordinator
    assert "event.data.resumed" in coordinator
    assert "next.splice(index, 1)" in coordinator
    assert 'event.event === "run.completed"' in coordinator
    assert "compactReplayUpdates" in coordinator
    assert "hydrateResearchReplay" in coordinator
    assert "replayThroughSeq" in coordinator
    assert "needsCatchup" in source("features/chat/api/research-api.ts")
    assert "Restoring research activity" in activity
    assert "useLayoutEffect" in activity
    assert "CollapsibleTrigger" in activity
    assert "activity.sources?.map" in activity
    assert "activityOpenByRunId" in coordinator
    assert "initializeActivityOpenState" not in coordinator
    assert "setActivityOpen(runId, activity.id, nextOpen)" in activity
    assert "open={open}" in activity
    assert "planReviewByRunId" in coordinator
    assert "setPlanReviewDraft" in coordinator
    assert "useResearchActivityScroll" in activity
    assert "MutationObserver" in activity
    assert "[overflow-anchor:none]" in activity
    assert 'behavior: "smooth"' not in activity
    assert "collapsible={showArtifactPanel}" in page
    assert "!artifactLayoutActive &&" in page
    assert '? "30%"' in page
    assert '? "58%"' in page
    assert "key={openResearchRunId}" in page
    assert "effectiveDeepResearchEnabled ? (" in thread
    assert "replayFrom: session?.lastAppliedSeq ?? 0" in coordinator
    assert "loadBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false)" in store
    checkpoint_update = store.split("setCheckpoint: (modelId, ggufVariant, options) =>", 1)[
        1
    ].split("setActiveThreadId:", 1)[0]
    assert "saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false)" in checkpoint_update
    # #8686 put a chat-scoped override in front of the global read here, so the literal `const permissionMode =
    # loadPermissionMode();` this used to pin is gone. The read itself is the contract, and it is still per call:
    # toggling deep research re-resolves the permission level, taking the chat's own level when it has one and the
    # persisted global otherwise, rather than reusing a stale value. Scoped to the setter, because over the whole file
    # this would also match the initial-state constant, which is a different property and would keep passing if this
    # read were dropped.
    deep_research_update = store.split("setDeepResearchEnabled: (deepResearchEnabled) =>", 1)[
        1
    ].split("setResearchWebsitePolicy:", 1)[0]
    assert re.search(
        r"const\s+permissionMode\s*=\s*threadScopedOverride\(\s*[\"']permissionMode[\"']\s*\)"
        r"\s*\?\?\s*loadPermissionMode\(\)",
        deep_research_update,
    ), (
        "toggling deep research must re-resolve permissionMode from the chat's own level "
        "falling back to the persisted global"
    )
    assert "permissionMode," in deep_research_update


def test_research_plan_and_status_contract() -> None:
    types = source("features/chat/types/research.ts")
    assert '| "queued"' in types
    assert '| "cancelling"' in types
    assert "title: string;" in types
    assert "query: string;" in types
    assert "position: number;" in types
    assert "createdAt: number;" in types
    assert "planRevision: number;" in types
    assert "planHash: string | null;" in types


def test_research_website_limits_are_configurable_and_sent_with_each_run() -> None:
    component = source("features/chat/components/deep-research-composer-button.tsx")
    thread = source("components/assistant-ui/thread.tsx")
    store = source("features/chat/stores/chat-runtime-store.ts")
    adapter = source("features/chat/api/chat-adapter.ts")

    assert 'label="Allow only"' in component
    assert 'label="Always block"' in component
    assert "their subdomains" in component
    assert "<DialogTitle>Deep research</DialogTitle>" in component
    assert "DeepResearchWebsiteAccessDialog" in thread
    assert "researchWebsitePolicy" in store
    assert "CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY" in store
    assert "websitePolicy:" in adapter
    assert "allowedDomains" in adapter and "blockedDomains" in adapter


def test_research_is_one_shot_per_thread_without_disabling_normal_chat() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    runtime = source("features/chat/runtime-provider.tsx")
    thread = source("components/assistant-ui/thread.tsx")
    coordinator = source("features/chat/stores/research-run-store.ts")

    assert "claimedThreadIds" in coordinator
    assert "setThreadClaimed" in coordinator
    assert "researchThreadState.hasRun" in runtime
    assert "threadAlreadyResearched" in adapter
    assert "runtime.setDeepResearchEnabled(false)" in adapter
    assert "effectiveDeepResearchEnabled" in thread
    assert "researchAvailable={!researchUsed}" in thread
    assert "{researchAvailable ? (" in thread
    assert "setToolsEnabled" in thread
    assert "Web search" in thread


def test_settled_terminal_research_never_stays_disconnected() -> None:
    coordinator = source("features/chat/stores/research-run-store.ts")
    activity = source("features/chat/components/research-activity-panel.tsx")

    assert "function isSettledResearchRun" in coordinator
    assert 'connection: settled ? "idle"' in coordinator
    assert "error: settled ? null" in coordinator
    assert 'state.setFollowing(runId, false, "idle")' in coordinator
    assert "!isSettledResearchRun(run, session.lastAppliedSeq)" in activity


def test_replayed_history_never_borrows_another_attempts_step_result() -> None:
    # A retry deletes the previous attempt's research_plan_steps rows but keeps its events, and the SSE route attaches
    # the live run snapshot to every replayed event. Matching a replayed step only by position would show the newest
    # attempt's evidence inside the older one.
    coordinator = source("features/chat/stores/research-run-store.ts")

    assert "const snapshotIsSameAttempt = attempt === (event.run.retryCount ?? 0);" in coordinator
    assert "const snapshot = snapshotIsSameAttempt" in coordinator
    assert "? event.run.steps.find((step) => step.position === stepPosition)" in coordinator
    assert "snapshot?.result?.evidenceSources ?? activity.evidenceSources," in coordinator
    assert "excerpt: snapshot?.result?.excerpt ?? activity.excerpt," in coordinator
    resumed_gate = coordinator.split('event.event === "run.started" &&', 1)[1].split("{", 1)[0]
    assert "event.data.resumed" in resumed_gate
    assert "snapshotIsSameAttempt" in resumed_gate


def test_research_stop_is_prompt_only_and_deduplicated() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    thread = source("components/assistant-ui/thread.tsx")
    activity = source("features/chat/components/research-activity-panel.tsx")

    assert "stoppingResearchRunIdRef" in thread
    # The composer selects the run status, not the run object, so a streamed research delta
    # does not re-render it. The cancelling guard reads that status.
    assert 'activeResearchRunStatus === "cancelling"' in thread
    assert 'aria-label={researchStopping ? "Stopping research"' in thread
    assert "cancelResearchRun" not in activity
    assert "Stop research" not in activity
    assert "abortSignal.reason as { detach?: boolean }" in adapter
    assert "await cancelResearchRun(createdRun.id)" in adapter


def test_the_handoff_is_keyed_on_the_result_the_backend_writes() -> None:
    """tool_end alone closes a denied, skipped or budget-exhausted call too.

    Reading a run out of one spends the chat's single Deep Research on a question the loop
    refused to pass on, so the two sides have to agree on what "it ran" looks like.
    """
    helper = source("features/chat/utils/deep-research-handoff.ts")
    tools = (ROOT / "studio" / "backend" / "core" / "inference" / "tools.py").read_text(
        encoding = "utf-8"
    )
    marker = re.search(r'DEEP_RESEARCH_STARTED_MARKER = "([^"]+)"', tools)
    assert marker is not None
    assert f'DEEP_RESEARCH_STARTED_MARKER = "{marker.group(1)}"' in helper
    assert "result.startsWith(DEEP_RESEARCH_STARTED_MARKER)" in helper
    # A gated call keeps its Allow / Deny card: the loop blocks on a verdict, so hiding it
    # asks the user nothing and the turn hangs there.
    assert "if (event.awaiting_confirmation === true) {\n      return false;" in helper

    # The clamp is the endpoint's own limit;
    # a longer question 422s the whole handoff.
    routes = (ROOT / "studio" / "backend" / "routes" / "research_runs.py").read_text(
        encoding = "utf-8"
    )
    max_length = re.search(
        r"question: str \| None = Field\(default = None, max_length = ([\d_]+)\)", routes
    )
    assert max_length is not None
    limit = int(max_length.group(1).replace("_", ""))
    assert f"DEEP_RESEARCH_QUESTION_MAX_CHARS = {limit}" in helper


def test_the_handoff_uses_the_turn_that_asked_for_it() -> None:
    """The handoff runs after the model's stream, and the model selector stays usable.

    Reading the store there researched model A's handoff with whichever model B had since
    been picked, or failed outright when B runs no Studio tools.
    """
    adapter = source("features/chat/api/chat-adapter.ts")
    # The generator's own body: the main path's re-read after an auto-load is a different
    # thing, and it happens at send time where reading the store is right.
    research = adapter.split("const startDeepResearch = async function*", 1)[1].split(
        "const deepResearchHandoff = newDeepResearchHandoff();", 1
    )[0]
    assert "const sendTimeRuntime = runtime;" in research
    assert "withResolvedModel(sendTimeRuntime)" in research
    # Only the empty-model resolution, which happens inside this call, may override it.
    assert "useChatRuntimeStore.getState()" not in research


def test_a_local_model_without_tools_cannot_consume_research_without_classifying() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    fallback = adapter.split("if (\n        deepResearchArmed &&", 1)[1].split(
        "      const ragProjectId = await resolveProjectId(", 1
    )[0]

    assert "!supportsTools" in fallback
    assert "!isExternalModelId(params.checkpoint)" in fallback
    assert "deepResearchArmed = false" in fallback
    assert "runtime.setDeepResearchEnabled(false)" in fallback
    assert "yield* startDeepResearch" not in fallback


def test_a_tool_that_ran_starts_its_run_even_if_the_reply_never_finished() -> None:
    """The acknowledgement can error, time out or drop; the run it announced is still owed."""
    adapter = source("features/chat/api/chat-adapter.ts")
    stream_error = adapter.split("} catch (streamError) {", 1)[1].split("throw streamError;", 1)[0]
    assert "deepResearchHandoff.question !== null && !runSignal.aborted" in stream_error
    assert "yield* startDeepResearch(deepResearchHandoff.question, runSignal)" in stream_error
    assert "if (deepResearchHandoff.question !== null) break;" in adapter
