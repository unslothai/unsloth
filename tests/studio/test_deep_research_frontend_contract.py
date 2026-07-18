from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend" / "src"


def source(path: str) -> str:
    return (FRONTEND / path).read_text(encoding = "utf-8")


def test_research_api_is_isolated_and_cursor_based() -> None:
    api = source("features/chat/api/research-api.ts")
    assert 'authFetch("/api/chat/research-runs"' in api
    assert "authFetch(`/api/chat/research-runs/active?${query}`)" in api
    assert "const { runs, hasRun }" in api
    assert "runs.at(-1) ?? null" in api
    assert "getResearchThreadState" in api
    assert "/events?after=${Math.max(0, after)}" in api
    assert 'headers: { accept: "text/event-stream" }' in api
    assert "export async function* followResearchRun" in api
    assert "Math.min(8_000, 500 * 2 ** (failures - 1))" in api
    assert "for await (const event of streamResearchEvents" in api
    assert 'source: "event"' in api
    assert "fresh.report !== run.report" in api
    assert "await waitForReconnect(" in api
    assert "while (!(run || signal?.aborted))" in api
    assert "isPermanentResearchError(error)" in api
    assert 'yield { run, source: "snapshot" }' in api
    for action in ("cancel", "retry"):
        assert f'mutate(id, "{action}")' in api
    assert 'mutate(id, "approve", { planRevision, planHash })' in api
    assert "JSON.stringify({ plan, expectedRevision })" in api


def test_research_mode_is_single_chat_and_detaches_without_cancel() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    assert "runtime.deepResearchEnabled" in adapter
    assert "!options.pairId" in adapter
    assert 'options.modelType === "base"' in adapter
    assert "cancelResearchRun(run.id)" not in adapter
    assert "createResearchRun" in adapter
    assert "await saveStoredChatMessage({" in adapter
    assert "unstable_assistantMessageId," in adapter
    assert "if (!unstable_assistantMessageId)" in adapter
    assert "assistantMessageId: unstable_assistantMessageId" in adapter
    assert "followResearchRun(createdRun.id" in adapter
    assert "inferenceRequest" in adapter
    assert "Number.isFinite(params.temperature)" in adapter
    assert "Number.isFinite(params.topP)" in adapter
    assert "Number.isFinite(params.maxTokens)" in adapter
    assert "Math.min(8192, Math.floor(params.maxTokens))" in adapter
    assert 'update.event?.event === "report.updated"' in adapter
    assert 'update.event?.event === "reasoning.updated"' in adapter
    assert "The activity store coalesces these high-frequency events" in adapter
    assert '{ type: "text" as const, text: report }' in adapter
    assert "if (abortSignal.aborted) return" in adapter
    assert "await autoLoadSmallestModel()" in adapter
    assert "signal: researchFollowController.signal" in adapter
    assert "beginExternalResearchFollow(" in adapter
    assert "ragScope" in adapter
    assert "runtime.ragEnabled\n                    ? { thread_id: resolvedThreadId }" in adapter
    create_block = adapter.split("createdRun = await createResearchRun({", 1)[1].split("});", 1)[0]
    assert "modelId:" not in create_block
    assert "prompt," not in create_block


def test_research_metadata_and_server_merge_are_persisted() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    runtime = source("features/chat/runtime-provider.tsx")
    assert "researchRunId: run.id" in adapter
    assert "serverManaged: true" in adapter
    assert "getResearchThreadState(remoteId)" in runtime
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
    assert "if (isResearchMessage) return null" in thread
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
    assert "effectiveDeepResearchEnabled ||" in thread
    assert "replayFrom: session?.lastAppliedSeq ?? 0" in coordinator
    assert "loadBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false)" in store
    checkpoint_update = store.split("setCheckpoint: (modelId, ggufVariant) =>", 1)[1].split(
        "setActiveThreadId:", 1
    )[0]
    assert "saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false)" in checkpoint_update
    assert "const permissionMode = loadPermissionMode();" in store
    assert "permissionMode," in store


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
    assert ">Websites</span>" in component
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


def test_research_stop_is_prompt_only_and_deduplicated() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    thread = source("components/assistant-ui/thread.tsx")
    activity = source("features/chat/components/research-activity-panel.tsx")

    assert "stoppingResearchRunIdRef" in thread
    assert 'activeResearchRun.status === "cancelling"' in thread
    assert 'aria-label={researchStopping ? "Stopping research"' in thread
    assert "cancelResearchRun" not in activity
    assert "Stop research" not in activity
    assert "abortSignal.reason as { detach?: boolean }" in adapter
    assert "await cancelResearchRun(createdRun.id)" in adapter
