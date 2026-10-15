# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A /chat?new=<uuid> view has no other recount trigger; a reload is retried by the dependency array."""

from __future__ import annotations

import json
import re
import textwrap

import pytest

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

REFRESH = source_path("studio/frontend/src/features/chat/utils/refresh-context-usage.ts")
PROVIDER = source_path("studio/frontend/src/features/chat/runtime-provider.tsx")
STORE = source_path("studio/frontend/src/features/chat/stores/chat-runtime-store.ts")
RUNTIME = source_path("studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts")
MESSAGE_ORDER = source_path("studio/frontend/src/features/chat/utils/message-order.ts")

TEMP = WORKDIR / "temp" / "new_chat_context_recount"

SOURCES = (REFRESH, PROVIDER, STORE, RUNTIME, MESSAGE_ORDER)

BOUND_NAMES = {
    "activeThreadId",
    "aui",
    "checkpoint",
    "enabled",
    "loadedContextLength",
    "isLoading",
    "mainThreadId",
    "runActive",
    "modelLoading",
    "newThreadSwitchStateRef",
    "nonce",
    "paused",
    "tipId",
    "tipRunning",
    "branchLength",
    "branchRevision",
}


def _refresh_module_body() -> str:
    """Body after the chat-history-storage import, minus later imports, which harness.ts cannot resolve."""
    text = read(REFRESH)
    marker = 'from "./chat-history-storage";'
    rest = text[text.index(marker) + len(marker) :]
    lines = rest.split("\n")
    while lines and (not lines[0].strip() or lines[0].startswith("import ")):
        lines.pop(0)
    return "\n" + "\n".join(lines)


def _message_order_body() -> str:
    """The replayed body must define orderBySelectedBranch, or the recount prices the wrong branch."""
    return read(MESSAGE_ORDER)


def _component_effects(start: str, end: str) -> list[tuple[list[str], str]]:
    """Every ``useEffect`` in one component as (dependency names, verbatim body)."""
    component = slice_between(read(PROVIDER), start, end)
    matches = re.findall(r"useEffect\(\(\) => \{\n(.*?)\n  \}, \[([^\]]*)\]\);", component, re.S)
    assert matches, f"{start} effects not found"
    effects = [
        ([name.strip() for name in deps.split(",") if name.strip()], body) for body, deps in matches
    ]
    for deps, _body in effects:
        unknown = set(deps) - BOUND_NAMES
        assert not unknown, f"emulator does not bind {sorted(unknown)}"
    return effects


def _new_chat_effects() -> list[tuple[list[str], str]]:
    return _component_effects("function ThreadNewChatSwitch(", "\nfunction ActiveThreadSync(")


def _thread_recount_effects() -> list[tuple[list[str], str]]:
    return _component_effects(
        "function ThreadContextUsageRecount(",
        "\nfunction CancelRegistrar(",
    )


def _store_reducers() -> str:
    """The four reducers the recount goes through, verbatim."""
    text = read(STORE)
    checkpoint = slice_between(
        text,
        "setCheckpoint: (modelId, ggufVariant, options) =>",
        "  setActiveThreadId: (activeThreadId) =>",
    )
    active = slice_between(
        text, "setActiveThreadId: (activeThreadId) =>", "applyThreadScopedSettings:"
    )
    usage = slice_between(text, "setContextUsage: (contextUsage) =>", "setThreadContextUsage:")
    thread_usage = slice_between(text, "setThreadContextUsage: (threadId, usage) =>", "}));")
    return (
        "  "
        + checkpoint.strip()
        + "\n  "
        + active.strip()
        + "\n  "
        + usage.strip()
        + "\n  "
        + thread_usage.strip()
    )


def _status_poll_adoption_tail() -> str:
    """The status sync's setCheckpoint-plus-recount tail, verbatim (API / CLI loads hydrate here)."""
    return slice_between(
        read(RUNTIME),
        "        setCheckpoint(checkpointId, statusRes.gguf_variant);",
        "      }\n    } else if (\n      !chatActiveModel &&",
    )


def _resident_fast_path() -> str:
    """Slices only the adoption tail: the residency decision imports 17 collaborators, so it is stubbed."""
    return slice_between(
        read(RUNTIME),
        "          const confirmedStatus = await readPickStatus();",
        # `lifecycleLease` is the first line about the load this tail does not start.
        "      let lifecycleLease: ModelLifecycleLease | null = null;",
    )


def _history_usage_restore() -> str:
    """The history loader's saved-usage restore and its recount call, verbatim."""
    return slice_between(
        read(PROVIDER),
        "        const checkpoint = store.params.checkpoint;",
        "        const hasParentIds = msgs.some((m) => m.parentId != null);",
    )


HARNESS_PRELUDE = """
// @ts-nocheck
// Fixtures the sliced source reads through. Everything below the PRELUDE marker
// is copied verbatim out of the studio sources.
export const world: any = {
  storedMessages: {} as Record<string, any[]>,
  countedMessages: [] as any[][],
  countedModel: undefined as string | undefined,
  switchedToNewThread: 0,
  clearedAttachments: 0,
  promptQueueStops: 0,
  // Set to { value: x } to stand in for a non-conforming 200 on the count path. Wrapped
  // so that { value: undefined } means "the reply omits input_tokens" rather than "no
  // override", which a bare undefined cannot express.
  countedTokensOverride: undefined as { value: any } | undefined,
  // Holds the count in flight, so a test can move the world while it is awaiting.
  countGate: null as Promise<void> | null,
};

const state: any = {
  activeThreadId: null,
  contextUsage: null,
  contextUsageByThreadId: {},
  params: { checkpoint: "", systemPrompt: "", systemVariables: "", maxTokens: 4096 },
  activeGgufVariant: null,
  loadedContextLength: null,
  modelLoading: false,
  runningByThreadId: {},
  // The subset decoding on the local llama-server: the recount must not share it with a decode.
  localRunByThreadId: {},
  // What the recount reads to tell an output-only audio GGUF from a chat one.
  models: [],
  // Deep Research sends a research run instead of this history, so the count would price
  // a request that is never made.
  deepResearchEnabled: false,
  // Per-model parameter memory, off and empty: the recount is about the window, not
  // about which model last set the temperature.
  rememberParamsPerModel: false,
  paramsByModel: {},
};

function set(updater: any): void {
  const patch = typeof updater === "function" ? updater(state) : updater;
  Object.assign(state, patch);
}

// What the sliced setCheckpoint reducer reads through: nothing is persisted here, and
// no external provider is configured, so its output-cap clamp is a no-op.
const CHAT_DEEP_RESEARCH_ENABLED_KEY = "unsloth_deep_research_enabled";
// The per-model memory, with the toggle off above: nothing is snapshotted, no map is
// written back, and a switch keeps the params on screen. Only the loaded-context cap
// is kept real, since that one clamps a value the recount prices.
function rememberOutgoingModel(_state: any, _outgoing: any): any {
  return null;
}
function getReplayedParams(
  _enabled: any,
  _byModel: any,
  current: any,
  _modelId: string,
  _checkpointChanged: boolean,
  maxTokensCap?: number,
): any {
  return maxTokensCap !== undefined && current.maxTokens > maxTokensCap
    ? { ...current, maxTokens: maxTokensCap }
    : current;
}
function getReplayStatePatch(): any {
  return {};
}
// The open chat's own sampling, laid back over the replay by the sliced
// setCheckpoint. No thread-scoped snapshot is ever seeded here -- this file is
// about the context window, not about which chat pinned a temperature -- so
// nothing is held and the replay passes through untouched. The real restore is
// covered by studio/frontend/tests/thread-scoped-pairing-invariants.test.ts.
function restoreThreadScopedParams(params: any): any {
  return params;
}
function saveLastExternalCheckpoint(_id: string | null): void {}
function saveBool(_key: string, _value: boolean): void {}
function parseExternalModelId(id: string): any {
  const [providerId, ...rest] = id.split(":");
  return rest.length > 0 ? { providerId, modelId: rest.join(":") } : null;
}
const useExternalProvidersStore: any = { getState: () => ({ providers: [] }) };
function getExternalMaxOutputTokens(
  _providerType: any,
  _modelId: any,
  _maxOutputTokens?: any,
): number {
  return 8192;
}
function shouldAdvanceQueuedSettingsEpoch(
  currentValues: Readonly<object>,
  nextValues: Readonly<object>,
  trackQueuedSettings = true,
): boolean {
  if (!trackQueuedSettings) return false;
  const keys = new Set([...Object.keys(currentValues), ...Object.keys(nextValues)]);
  return Array.from(keys).some(
    (key) => !Object.is((currentValues as any)[key], (nextValues as any)[key]),
  );
}

const actions: any = {
__STORE_REDUCERS__
  // Not the real reducer. The adoption tail calls this only to restore the outgoing
  // maxSeqLength cap, which this file does not measure, while the real one reaches
  // preset policy, per-turn counters and loaded-context bookkeeping -- a web of
  // collaborators that would have to be stubbed to replay a merge. The merge is what
  // the tail depends on, so the merge is what this does.
  setParams: (params: any) =>
    set((current: any) => ({ params: { ...current.params, ...params } })),
};

export const useChatRuntimeStore: any = {
  getState: () => ({ ...state, ...actions }),
  // The sliced load paths write straight to the store as zustand allows, so the fake needs the
  // same door. Merge, like the real setState: a replacing one would drop the seeded fixture.
  setState: (patch: any) =>
    Object.assign(state, typeof patch === "function" ? patch(state) : patch),
};

export function seed(patch: any): void {
  Object.assign(state, patch);
}

export function snapshot(): any {
  return {
    activeThreadId: state.activeThreadId,
    contextUsage: state.contextUsage,
    contextUsageByThreadId: state.contextUsageByThreadId,
  };
}

async function listStoredChatMessages(id: string): Promise<any[]> {
  return world.storedMessages[id] ?? [];
}

function isExternalModelId(id: string): boolean {
  return typeof id === "string" && id.includes(":");
}

function findLatestUserAudioBase64(_messages: any): string | null {
  return null;
}

// refresh-context-usage.ts declines to price a prompt carrying video, the same way
// it declines audio and images. The emulator replays that module's body with its
// imports stripped, so every name it imports has to exist here or the bail throws
// a ReferenceError and the recount never runs -- which is what happened: adding
// this import took `counts` to 0 and read as "the empty New Chat view must be
// priced exactly once" failing on a pricing bug. See
// test_the_harness_stubs_every_name_refresh_context_usage_imports.
function findLatestUserVideoBase64(_messages: any): string | null {
  return null;
}

// refresh-context-usage.ts declines to price a prompt carrying video, the same way
// it declines audio and images. The emulator replays that module's body with its
// imports stripped, so every name it imports has to exist here or the bail throws
// a ReferenceError and the recount never runs -- which is what happened: adding
// this import took `counts` to 0 and read as "the empty New Chat view must be
// priced exactly once" failing on a pricing bug. See
// test_the_harness_stubs_every_name_refresh_context_usage_imports.


// The real predicate's rule, so a test can put an image on a branch and see it declined.
function messagesContainImage(messages: any): boolean {
  const isImage = (p: any) => p?.type === "image" && Boolean(p?.image);
  for (const message of messages) {
    for (const part of message.content ?? []) if (isImage(part)) return true;
    for (const attachment of message.attachments ?? []) {
      for (const part of attachment.content ?? []) if (isImage(part)) return true;
    }
  }
  return false;
}

// The adapter's own prompt build is exercised by the request tests; here it only has
// to turn the reconstructed branch into something countable. Same shape
// refreshContextUsage spreads into countChatInputTokens.
async function buildLocalTokenCountHistory(messages: any): Promise<{ messages: any[] }> {
  return { messages: messages.map((m: any) => ({ role: m.role, content: "x" })) };
}

async function buildLocalTokenCountExtras(): Promise<Record<string, unknown>> {
  return {};
}

// The reasoning kwargs are pinned by test_token_count_prompt_parity.py; here the count
// only has to reach the server with the right branch of messages.
function buildLocalTokenCountReasoning(): Record<string, unknown> {
  return {};
}

// 12 tokens for the bare template, 25 more per message actually sent. The endpoint also
// names the tokenizer that produced the total; world.countedModel unset means the reply
// omits it, as an older backend would.
async function countChatInputTokens(payload: any): Promise<any> {
  world.countedMessages.push(payload.messages);
  if (world.countGate) await world.countGate;
  return {
    input_tokens:
      world.countedTokensOverride === undefined
        ? 12 + payload.messages.length * 25
        : world.countedTokensOverride.value,
    ...(world.countedModel === undefined ? {} : { model: world.countedModel }),
  };
}

const requestPromptQueueStop = (_opts: any): void => {
  world.promptQueueStops += 1;
};
const requestTemporaryPromptQueueStop = (): void => {
  world.promptQueueStops += 1;
};

const auiFixture: any = {
  threads: () => ({
    switchToNewThread: async () => {
      world.switchedToNewThread += 1;
    },
    // The runtime's main thread follows the store unless a test pins another one.
    getState: () => ({ mainThreadId: world.mainThreadId ?? state.activeThreadId }),
  }),
  // The mounted main thread's visible branch: a test's live messages, else its stored records.
  thread: () => ({
    getState: () => ({
      messages:
        world.liveMessages?.[state.activeThreadId] ??
        (world.storedMessages[state.activeThreadId] ?? []).map(storedMessageToRunMessage),
    }),
  }),
  // The switch clears a staged attachment before moving on, so the composer has to
  // exist here: a missing one throws inside the effect and the recount below it
  // never runs, which reads as a pricing bug rather than a missing stub.
  composer: () => ({
    clearAttachments: async () => {
      world.clearedAttachments += 1;
    },
  }),
};

// ---- PRELUDE ENDS: verbatim studio source follows ----
"""

HARNESS_RENDER = """

// Replays sliced effects with React's dependency rule: an effect re-runs only when one
// of its own dependencies changed since the last render.
function replayEffects(effects: any[], scope: any, memo: any[]): void {
  effects.forEach((effect: any, index: number) => {
    const next = effect.deps.map((name: string) => scope[name]);
    const previous = memo[index];
    if (
      previous != null &&
      previous.length === next.length &&
      previous.every((value: any, i: number) => Object.is(value, next[i]))
    ) {
      return;
    }
    memo[index] = next;
    effect.run();
  });
}

const recountDeps: any[] = [];
const pricedTipRef: any = { current: null };

export function renderThreadContextUsageRecount(props: any = {}): void {
  const enabled = props.enabled ?? true;
  const aui = auiFixture;
  // Read through the store the way the component's selectors do.
  const activeThreadId = state.activeThreadId;
  const checkpoint = state.params.checkpoint;
  const loadedContextLength = state.loadedContextLength;
  const modelLoading = state.modelLoading;
  const runActive = Object.values(state.runningByThreadId ?? {}).some(Boolean);
  const messages = aui.thread().getState().messages;
  const tip = messages.at(-1);
  const tipId = tip?.id;
  const tipRunning = tip?.role === "assistant" && tip.status?.type === "running";
  const branchLength = messages.length;
  const branchRevision = revisionTotal(messages);
  const scope: any = {
    activeThreadId,
    aui,
    checkpoint,
    enabled,
    loadedContextLength,
    modelLoading,
    runActive,
    tipId,
    tipRunning,
    branchLength,
    branchRevision,
  };
  const effects: any[] = [
__RECOUNT_EFFECTS__
  ];
  replayEffects(effects, scope, recountDeps);
}

const renderedDeps: any[] = [];
const newThreadSwitchStateRef: any = {
  // attempt mirrors the real ref: the effect reads `attempt + 1`, so omitting it makes
  // every attempt NaN and NaN !== NaN skips the deferred clear the switch armed.
  current: { activeNonce: null, hasSwitched: false, attempt: 0, pendingSavedThreadIds: [] },
};

export function leaveNewChatForSavedThread(): void {
  newThreadSwitchStateRef.current.activeNonce = null;
  renderedDeps.length = 0;
}

export function markImplicitNewChatUsed(): void {
  newThreadSwitchStateRef.current.hasSwitched = true;
}

export function renderNewChatSwitch(props: any): void {
  const aui = auiFixture;
  const isLoading = props.isLoading;
  const nonce = props.nonce;
  // Compare keeps the shared provider mounted but stood down; the recount tests are
  // all about the view the user is looking at, so it defaults to on screen.
  const paused = props.paused ?? false;
  // The stale-switch correction reads it. Defaulting to a runtime-made id keeps that
  // effect inert here: these tests are about the recount, and a local id is what a
  // `?new=` view actually holds.
  const mainThreadId = props.mainThreadId ?? "__LOCALID_recount";

  // The component reads these through useChatRuntimeStore selectors, so a
  // re-render sees whatever the store holds right now.
  const checkpoint = state.params.checkpoint;
  const loadedContextLength = state.loadedContextLength;
  const modelLoading = state.modelLoading;
  const runActive = Object.values(state.runningByThreadId ?? {}).some(Boolean);
  const scope: any = {
    aui,
    isLoading,
    mainThreadId,
    newThreadSwitchStateRef,
    nonce,
    paused,
    checkpoint,
    loadedContextLength,
    modelLoading,
    runActive,
  };
  const effects: any[] = [
__EFFECTS__
  ];
  replayEffects(effects, scope, renderedDeps);
}
"""


HARNESS_RESIDENT = """

// Adopting a model that never left memory returns before the post-load recount, so the
// tail is sliced verbatim below and replayed with the load machinery stubbed. Each stub
// is either derived as the source derives it, or a collaborator this file does not
// measure.
export async function adoptResidentModel(props: any): Promise<void> {
  const selection = "pick";
  const modelId: string = props.modelId;
  const loadPath: string = props.modelId;
  const selectedCheckpoint: string | null = state.params?.checkpoint ?? null;
  const previousGgufVariant: string | null = state.activeGgufVariant ?? null;
  const pendingConfig: any = undefined;
  // The residency decision, which this file does not measure: the caller seeds the
  // status it wants adopted. resident-model-match.test.ts covers the real predicate.
  const adoptable = (_status: any): boolean => true;
  // The superseded-pick guards the tail sits behind: nothing here supersedes the pick, so the
  // epoch still matches the id it was minted with and no rival holds the picker entry.
  let modelSelectionIntentEpoch = 0;
  const loadIntentId = 0;
  const rivalLoadStarted = (): boolean => false;
  let pendingReplacementRollback: any = null;
  const restorePreviousConfig = (): void => {};
  const getInferenceStatus = async (): Promise<any> => props.residentStatus;
  const readPickStatus = async (): Promise<any> => props.residentStatus;
  const reconcilePersistedGpuIds = (ids: any): any => ids;
  const sameGpuSelection = (a: any, b: any): boolean =>
    JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
  // The real hydration writes the whole status; the recount only reads the window.
  const applyActiveModelStatusToStore = (status: any, _options: any): void => {
    set({
      loadedContextLength: status.is_gguf ? (status.context_length ?? null) : null,
    });
  };
  const syncModelCapabilities = (_id: string, _status: any): void => {};
  const applyPerModelConfigToRuntime = (_config: any, _options?: any): void => {};
  // Only maxSeqLength is read, and only to decide whether the pick names a cap. The
  // real normalizer also snaps and clamps; what matters here is null-for-absent.
  const normalizeMaxSeqLength = (value: unknown): number | null =>
    typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
  const defaultInferenceParams: any = { maxSeqLength: null };
  const forceReload = false;
  const nativePathToken = undefined;
  const residentStatus: any = props.residentStatus;
  const readServerWideReloadHints = async (): Promise<boolean> => false;
  // The two conditions the tail sits inside, restated so the braces balance and the
  // entry conditions are visible. Both say the same thing: this pick needs no load.
  if (!forceReload && !nativePathToken) {
    if (
      residentStatus &&
      adoptable(residentStatus) &&
      !(await readServerWideReloadHints())
    ) {
__FAST_PATH__
}
"""


HARNESS_STATUS_POLL = """

export async function adoptServerStatus(statusRes: any): Promise<void> {
  const selectedCheckpoint: string = state.params?.checkpoint ?? "";
  const checkpointId: string | null =
    statusRes.model_identifier ?? statusRes.active_model ?? null;
  const previousGgufVariant: string | null = state.activeGgufVariant ?? null;
  const { setCheckpoint } = useChatRuntimeStore.getState();
  const applyActiveModelStatusToStore = (status: any, _options: any): void => {
    set({
      loadedContextLength: status.is_gguf ? (status.context_length ?? null) : null,
    });
  };
  const syncModelCapabilities = (_id: string, _status: any): void => {};
  if (checkpointId) {
__STATUS_POLL_TAIL__
  }
}
"""


def _rendered_effects(effects: list[tuple[list[str], str]]) -> str:
    blocks = []
    for deps, body in effects:
        blocks.append(
            "    {\n"
            f"      deps: {json.dumps(deps)},\n"
            "      run: () => {\n"
            f"{body}\n"
            "      },\n"
            "    },"
        )
    return "\n".join(blocks)


HARNESS_HISTORY = """

// Opening a stored thread runs this inside the history adapter, before assistant-ui is handed
// the repository, so the live branch is empty and only the stored path can answer.
export async function hydrateThreadUsage(props: any): Promise<void> {
  const remoteId: string = props.remoteId;
  // The loader is created per pane; a compare pane carries a pairId and never owns the bar.
  const modelType: string = props.modelType ?? "base";
  const pairId = props.pairId ?? undefined;
  // The thread's stored messages, which the loader reads into `msgs` and sorts above the sliced
  // block (createdAt, then role, then id), so `msgs.at(-1)` below is the newest stored row.
  const roleOrder: Record<string, number> = { system: 0, user: 1, assistant: 2 };
  const stored: any[] = props.messages ?? [];
  const savedReply = props.savedUsage
    ? [{
        id: "a-saved",
        parentId: stored.at(-1)?.id ?? null,
        role: "assistant",
        createdAt: (stored.at(-1)?.createdAt ?? 0) + 1,
        content: [{ type: "text", text: "" }],
        metadata: { contextUsage: props.savedUsage },
      }]
    : [];
  const msgs: any[] = [...stored, ...savedReply].sort((a: any, b: any) => {
    if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
    const aOrder = roleOrder[a.role] ?? 99;
    const bOrder = roleOrder[b.role] ?? 99;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  // The branch the loader selects from `msgs` above the sliced block, which the block prices with
  // `estimateContextUsage` when nothing saved is usable (#9475, #13109). No head is saved here, so
  // it follows the newest stored row to its leaf, as the loader does without one.
  const branch: any[] = orderBySelectedBranch(
    msgs,
    resolveSavedBranchHead(msgs, props.savedHeadId ?? msgs.at(-1)?.id),
  );
  // Read once, as the loader does, just above the sliced block.
  const store = useChatRuntimeStore.getState();
__RESTORE__
}
"""


def _harness_source() -> str:
    prelude = HARNESS_PRELUDE.replace("__STORE_REDUCERS__", _store_reducers())
    render = HARNESS_RENDER.replace("__EFFECTS__", _rendered_effects(_new_chat_effects())).replace(
        "__RECOUNT_EFFECTS__", _rendered_effects(_thread_recount_effects())
    )
    resident = HARNESS_RESIDENT.replace("__FAST_PATH__", _resident_fast_path())
    status_poll = HARNESS_STATUS_POLL.replace("__STATUS_POLL_TAIL__", _status_poll_adoption_tail())
    history = HARNESS_HISTORY.replace("__RESTORE__", _history_usage_restore())
    return (
        prelude
        + _message_order_body()
        + _refresh_module_body()
        + render
        + resident
        + status_poll
        + history
    )


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script, sources = SOURCES)


LOADED_MODEL = """
    seed({
      params: { checkpoint: "unsloth/gguf-model", systemPrompt: "", systemVariables: "" },
      loadedContextLength: 8192,
      modelLoading: false,
    });
"""


def test_the_harness_stubs_every_name_refresh_context_usage_imports() -> None:
    """Every chat-adapter name refresh-context-usage imports must be stubbed, or the recount stays 0."""
    text = read(REFRESH)
    block = re.search(r"import \{(.*?)\} from \"\.\./api/chat-adapter\";", text, re.S)
    assert block, "could not find the chat-adapter import block in refresh-context-usage.ts"
    imported = [
        name.strip()
        for name in block.group(1).split(",")
        if name.strip() and not name.strip().startswith("type ")
    ]
    assert imported, "parsed an empty import list; this guard would check nothing"

    with open(__file__, encoding = "utf-8") as handle:
        harness = handle.read()
    missing = [
        name
        for name in imported
        if f"function {name}(" not in harness and f"const {name} =" not in harness
    ]
    assert not missing, (
        f"refresh-context-usage.ts imports {missing}, which this harness does not "
        "define. The replayed module body would throw a ReferenceError and the "
        "recount would silently report 0. Add a stub next to "
        "findLatestUserAudioBase64."
    )


@pytest.mark.parametrize(
    ("before_status", "expected_early_counts"),
    [
        # The second render repeats identical store values and must not re-price.
        pytest.param(
            LOADED_MODEL
            + """
            seed({
              activeThreadId: "thread-a",
              contextUsage: { promptTokens: 900, completionTokens: 30, totalTokens: 930, cachedTokens: 0 },
            });
            """,
            1,
            id = "model_already_resident",
        ),
        pytest.param("", 0, id = "reload_before_status_hydrates"),
        # The previous runtime keeps returning its messages until switchToNewThread() settles.
        pytest.param(
            LOADED_MODEL
            + """
            seed({
              activeThreadId: "thread-a",
              contextUsage: { promptTokens: 900, completionTokens: 30, totalTokens: 930, cachedTokens: 0 },
            });
            setActiveBranchReader(() => [
              { id: "m1", role: "user", createdAt: new Date(1), content: [{ type: "text", text: "hi" }] },
              { id: "m2", role: "assistant", createdAt: new Date(2), content: [{ type: "text", text: "yo" }] },
            ]);
            """,
            1,
            id = "outgoing_conversation_still_mounted",
        ),
    ],
)
def test_a_new_chat_prices_its_empty_prompt_against_a_resident_gguf(
    before_status, expected_early_counts
):
    """#7450: the empty New Chat request already carries a template, but this view reaches no other
    recount trigger. It has to land on a count in either order, and open exactly one thread."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              renderNewChatSwitch,
              seed,
              setActiveBranchReader,
              snapshot,
              world,
            }} from "./harness.ts";
            {before_status}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            const early = {{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }};

            // /api/inference/status answers; a no-op when the model was already known.
            {LOADED_MODEL}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            const after = snapshot();
            console.log(JSON.stringify({{
              early,
              switched: world.switchedToNewThread,
              promptQueueStops: world.promptQueueStops,
              activeThreadId: after.activeThreadId,
              contextUsage: after.contextUsage,
              counts: world.countedMessages.length,
            }}));
            """
        )
    )
    assert out["early"].get("counts") == expected_early_counts
    if expected_early_counts == 0:
        assert out["early"].get("contextUsage") is None
    assert out["activeThreadId"] is None
    assert out["counts"] == 1, "the empty New Chat view must be priced exactly once"
    assert out["contextUsage"] is not None, (
        "without the recount the bar stays hidden on a New Chat opened against an "
        "already-resident GGUF, until the first completion"
    )
    assert out["contextUsage"].get("totalTokens") == 12
    assert out["contextUsage"].get("completionTokens") == 0
    assert out["switched"] == 1, "hydration must not open a second thread"
    assert out["promptQueueStops"] == 1, "hydration must not re-stop the prompt queue"


def test_a_backgrounded_new_chat_view_neither_opens_a_thread_nor_prices_one():
    """While paused (compare keeps the provider mounted), no thread is opened or priced until it lifts."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ activeThreadId: "thread-on-screen" }});

            renderNewChatSwitch({{ isLoading: false, nonce: "n1", paused: true }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            const paused = {{
              switched: world.switchedToNewThread,
              counts: world.countedMessages.length,
              activeThreadId: snapshot().activeThreadId,
              contextUsage: snapshot().contextUsage,
            }};

            // Compare closes: the view is back on screen and owes both.
            renderNewChatSwitch({{ isLoading: false, nonce: "n1", paused: false }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{
              paused,
              switched: world.switchedToNewThread,
              counts: world.countedMessages.length,
              activeThreadId: snapshot().activeThreadId,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["paused"]["switched"] == 0, "a paused switch must not open a thread"
    assert out["paused"]["counts"] == 0, "a paused switch must not price a prompt"
    assert (
        out["paused"]["activeThreadId"] == "thread-on-screen"
    ), "a paused switch must not blank the active thread the visible view is using"
    assert out["paused"]["contextUsage"] is None
    assert out["switched"] == 1, "releasing the pause must open the new thread"
    assert out["activeThreadId"] is None
    assert out["counts"] == 1, "releasing the pause must price the empty prompt once"
    assert out["contextUsage"] is not None


def test_a_staged_attachment_is_cleared_only_when_the_switch_moves_on():
    """switchToNewThread() reuses the composer, so an unsent attachment must not follow the user."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, world }} from "./harness.ts";
            {LOADED_MODEL}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            const first = {{
              switched: world.switchedToNewThread,
              cleared: world.clearedAttachments,
            }};

            // A re-render that changes nothing must not clear anything either.
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            const again = {{
              switched: world.switchedToNewThread,
              cleared: world.clearedAttachments,
            }};

            // New Chat, or the next project's landing: a different nonce.
            renderNewChatSwitch({{ isLoading: false, nonce: "n2" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{
              first,
              again,
              switched: world.switchedToNewThread,
              cleared: world.clearedAttachments,
            }}));
            """
        )
    )
    assert out["first"] == {
        "switched": 1,
        "cleared": 0,
    }, "the first switch has no outgoing composer to clear"
    assert out["again"] == {
        "switched": 1,
        "cleared": 0,
    }, "a re-render at the same nonce must not switch or clear again"
    assert out["switched"] == 2
    assert out["cleared"] == 1, "moving to another nonce must not carry the attachment"


def test_the_first_nonce_switch_clears_an_implicit_new_chat_attachment():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ markImplicitNewChatUsed, renderNewChatSwitch, seed, world }} from "./harness.ts";
            {LOADED_MODEL}
            markImplicitNewChatUsed();
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{
              switched: world.switchedToNewThread,
              cleared: world.clearedAttachments,
            }}));
            """
        )
    )
    assert out["switched"] == 1
    assert out["cleared"] == 1, "a staged implicit-chat attachment must not follow the nonce"


def test_back_to_the_same_new_chat_nonce_switches_after_a_saved_thread():
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ leaveNewChatForSavedThread, renderNewChatSwitch, seed, world }} from "./harness.ts";
            {LOADED_MODEL}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            leaveNewChatForSavedThread();
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{
              switched: world.switchedToNewThread,
              cleared: world.clearedAttachments,
            }}));
            """
        )
    )
    assert out["switched"] == 2, "Back to the same nonce must restore the new thread"
    assert out["cleared"] == 1, "the reused new-thread composer must lose its attachment"


@pytest.mark.parametrize(
    "reply",
    ["undefined", "null", '"1670"', "NaN", "Infinity", "{}"],
    ids = ["missing", "null", "string", "nan", "infinity", "object"],
)
def test_a_count_that_is_not_a_finite_number_never_reaches_the_bar(reply):
    """The response type is a compile-time assertion, so a 200 from anything but a matched backend
    can put a non-number on the bar. ContextUsageBar's "nothing to show" guard is `used <= 0`,
    which undefined does not satisfy, and its tooltip calls toLocaleString on it."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            world.countedTokensOverride = {{ value: {reply} }};
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1, "the count must still be attempted"
    assert out["contextUsage"] is None, (
        f"a reply of {reply} was published to the bar; it renders as "
        '"undefined / 8.2k" and throws from toLocaleString on hover'
    )


NO_LOCAL_MODEL = """
    seed({
      params: { checkpoint: "", systemPrompt: "", systemVariables: "" },
      loadedContextLength: null,
    });
"""


@pytest.mark.parametrize(
    ("seed_script", "is_loading", "expected_switched"),
    [
        pytest.param(NO_LOCAL_MODEL, "false", 1, id = "no_local_model"),
        pytest.param(LOADED_MODEL, "true", 0, id = "assistant_ui_still_loading"),
    ],
)
def test_the_bar_stays_hidden_when_there_is_nothing_to_price(
    seed_script, is_loading, expected_switched, request
):
    case = request.node.callspec.id
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            {seed_script}
            renderNewChatSwitch({{ isLoading: {is_loading}, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            console.log(JSON.stringify({{
              switched: world.switchedToNewThread,
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 0, case
    assert out["contextUsage"] is None, case
    assert out["switched"] == expected_switched, case


def test_the_hydration_retry_stays_off_a_started_chat():
    """The user sent a message before the status response landed, so a real completion owns the
    bar. The retry effect re-runs on any model field change, so it needs its own guard."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            // First turn completed against the resident model, on a persisted thread.
            {LOADED_MODEL}
            seed({{
              activeThreadId: "thread-a",
              contextUsage: {{ promptTokens: 640, completionTokens: 40, totalTokens: 680, cachedTokens: 0 }},
            }});
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 0, "a real completion's usage must not be recounted away"
    assert out["contextUsage"].get("totalTokens") == 680


TWO_STORED_TURNS = """
    world.storedMessages["thread-a"] = [
      { id: "m1", role: "user", createdAt: 1, content: [{ type: "text", text: "hi" }], metadata: {} },
      { id: "m2", role: "assistant", createdAt: 2, content: [{ type: "text", text: "yo" }], metadata: {} },
    ];
"""

RETRY_BRANCH_STORED = """
    world.storedMessages["thread-a"] = [
      { id: "m1", role: "user", createdAt: 1, content: [{ type: "text", text: "hi" }], metadata: {} },
      { id: "m2", role: "assistant", createdAt: 2, parentId: "m1", content: [{ type: "text", text: "yo" }], metadata: {} },
      { id: "m3", role: "user", createdAt: 3, parentId: "m2", content: [{ type: "text", text: "again" }], metadata: {} },
      { id: "m4", role: "assistant", createdAt: 4, parentId: "m3", content: [{ type: "text", text: "sure" }], metadata: {} },
    ];
"""

LIVE_BRANCH = """
    setActiveBranchReader(() => [
      { id: "m1", role: "user", createdAt: new Date(1), content: [{ type: "text", text: "hi" }] },
      { id: "m2", role: "assistant", createdAt: new Date(2), content: [{ type: "text", text: "yo" }] },
    ]);
"""

LIVE_INCOGNITO_BRANCH = """
    setActiveBranchReader(() => [
      { id: "m1", role: "user", createdAt: new Date(1), content: [{ type: "text", text: "hi" }] },
      { id: "m2", role: "assistant", createdAt: new Date(2), content: [{ type: "text", text: "yo" }] },
      { id: "m3", role: "user", createdAt: new Date(3), content: [{ type: "text", text: "more" }] },
    ]);
"""


@pytest.mark.parametrize(
    ("world_setup", "expected_sent", "counted_model"),
    [
        pytest.param(TWO_STORED_TURNS, 2, None, id = "stored_branch"),
        pytest.param(LIVE_INCOGNITO_BRANCH, 3, None, id = "incognito_thread_stores_nothing"),
        pytest.param(
            RETRY_BRANCH_STORED + LIVE_BRANCH, 2, None, id = "runtime_shows_an_older_branch"
        ),
        # The endpoint counts with whatever is resident; the reported id is the only witness.
        pytest.param(
            TWO_STORED_TURNS, 2, "unsloth/other-gguf", id = "another_client_swapped_the_model"
        ),
    ],
)
def test_a_loaded_model_reprices_the_open_thread(world_setup, expected_sent, counted_model):
    """Refills the per-thread cache after a model load by pricing the branch the next request would send."""
    expected_total = 12 + 25 * expected_sent if counted_model is None else None
    counted_model_setup = (
        "" if counted_model is None else f"world.countedModel = {json.dumps(counted_model)};"
    )
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage,
              seed,
              setActiveBranchReader,
              snapshot,
              world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            {counted_model_setup}
            {world_setup}
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});
            await refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});

            const after = snapshot();
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              sent: world.countedMessages.at(-1) ?? [],
              contextUsage: after.contextUsage,
              cached: after.contextUsageByThreadId["thread-a"] ?? null,
            }}));
            """
        )
    )
    assert out["counts"] == 1
    assert len(out["sent"]) == expected_sent, "the branch the request would send must be priced"
    assert (out["contextUsage"] or {}).get(
        "totalTokens"
    ) == expected_total, "a total from another model's tokenizer must not reach the bar"
    assert (out["cached"] or {}).get(
        "totalTokens"
    ) == expected_total, "nor the per-thread cache setActiveThreadId restores from"
    if expected_total is not None:
        assert out["cached"] is not None, "the recount must reach the per-thread cache"


@pytest.mark.parametrize(
    ("send_a_turn", "expected_total"),
    [
        pytest.param(True, None, id = "a_turn_arrives_mid_count"),
        pytest.param(False, 62, id = "branch_unchanged"),
    ],
)
def test_a_turn_sent_while_counting_drops_the_count(send_a_turn, expected_total):
    """The recount publishes for the branch it priced. A run that starts after the branch is fixed
    and is stopped before writing usage leaves contextUsage reference-equal to the snapshot, so the
    total lands anyway -- short by the turn just sent, until the next completion."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage,
              seed,
              setActiveBranchReader,
              snapshot,
              world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            const live = [
              {{ id: "m1", role: "user", createdAt: new Date(1), content: [{{ type: "text", text: "hi" }}] }},
              {{ id: "m2", role: "assistant", createdAt: new Date(2), content: [{{ type: "text", text: "yo" }}] }},
            ];
            setActiveBranchReader(() => live.slice());
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const pending = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));
            if ({str(send_a_turn).lower()}) {{
              // Sent, then stopped before the first token: no usage is ever written.
              live.push({{
                id: "m3",
                role: "user",
                createdAt: new Date(3),
                content: [{{ type: "text", text: "a very long pasted document" }}],
              }});
            }}
            release();
            await pending;

            const after = snapshot();
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              sent: world.countedMessages.at(-1) ?? [],
              contextUsage: after.contextUsage,
              cached: after.contextUsageByThreadId["thread-a"] ?? null,
            }}));
            """
        )
    )
    assert out["counts"] == 1
    assert len(out["sent"]) == 2, "the count priced the branch as it stood"
    assert (out["contextUsage"] or {}).get(
        "totalTokens"
    ) == expected_total, "a total for a branch that has since gained a turn must not reach the bar"
    assert (out["cached"] or {}).get("totalTokens") == expected_total


@pytest.mark.parametrize(
    ("running", "grew", "expected_total"),
    [
        # A run that begins after the count: only the publish guard can drop the total.
        (True, True, None),
        # Stopped before the count returned: only the content makes the branch look different.
        (False, True, None),
        (False, False, 62),
    ],
    ids = ["run_starts_mid_count", "stopped_before_publish", "idle_and_unchanged"],
)
def test_a_count_taken_while_the_thread_is_running_is_dropped(running, grew, expected_total):
    """Streaming grows a turn without moving its branch length or last id, so a mid-run count is dropped."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage,
              seed,
              setActiveBranchReader,
              snapshot,
              world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            const live = [
              {{ id: "m1", role: "user", createdAt: new Date(1), content: [{{ type: "text", text: "hi" }}] }},
              {{ id: "m2", role: "assistant", createdAt: new Date(2), content: [{{ type: "text", text: "yo" }}] }},
            ];
            setActiveBranchReader(() => live.slice());
            seed({{
              activeThreadId: "thread-a",
              contextUsage: null,
              contextUsageByThreadId: {{}},
              runningByThreadId: {{}},
            }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const pending = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));
            if ({str(running).lower()}) {{
              seed({{ runningByThreadId: {{ "thread-a": true }} }});
            }}
            if ({str(grew).lower()}) {{
              // Same id, same part count: only the streamed content grew.
              live[1].content = [{{ type: "text", text: "yo" + " and a great deal more" }}];
            }}
            release();
            await pending;

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1, "the count is still attempted"
    assert (out["contextUsage"] or {}).get("totalTokens") == expected_total, (
        "a partial mid-stream total must not reach the bar"
        if expected_total is None
        else "an idle, unchanged thread must still publish"
    )


@pytest.mark.parametrize(
    ("mutation", "expected_total"),
    [
        ("live[1].content[0] = { ...live[1].content[0], result: { rows: 4000 } };", None),
        # Same length, different text: a size-based signature cannot see it.
        ('live[0].content = [{ type: "text", text: "ih" }];', None),
        ("live[0].attachments = [];", None),
        ("", 62),
    ],
    ids = [
        "tool_result_filled_in",
        "text_swapped_same_length",
        "attachment_deleted",
        "unchanged",
    ],
)
def test_a_count_for_a_branch_mutated_without_growing_is_dropped(mutation, expected_total):
    """The branch a count priced has to be identified by content, not size: a tool loop mutates
    parts in place, so a size-based signature would publish a total for a stale prompt."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage,
              seed,
              setActiveBranchReader,
              snapshot,
              world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            const live = [
              {{
                id: "m1",
                role: "user",
                createdAt: new Date(1),
                content: [{{ type: "text", text: "hi" }}],
                attachments: [{{
                  id: "a1",
                  name: "notes.txt",
                  content: [{{ type: "text", text: "a stored note the count priced" }}],
                }}],
              }},
              {{
                id: "m2",
                role: "assistant",
                createdAt: new Date(2),
                content: [{{
                  type: "tool-call",
                  toolCallId: "c1",
                  toolName: "query_docs",
                  argsText: '{{"q":"hi"}}',
                  args: {{ q: "hi" }},
                }}],
              }},
            ];
            setActiveBranchReader(() => live.slice());
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const pending = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));
            {mutation}
            release();
            await pending;

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1, "the count is still attempted"
    assert (out["contextUsage"] or {}).get("totalTokens") == expected_total, (
        "a total for content that has since changed must not reach the bar"
        if expected_total is None
        else "an unchanged branch must still publish"
    )


@pytest.mark.parametrize(
    ("empties", "expected_total"),
    [(True, None), (False, 62)],
    ids = ["sole_exchange_deleted", "branch_kept"],
)
def test_a_count_for_a_branch_that_was_emptied_is_dropped(empties, expected_total):
    """Deleting the only exchange while a count is in flight does not touch contextUsage, so
    the old conversation's total would land on the now-empty thread."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage,
              seed,
              setActiveBranchReader,
              snapshot,
              world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            let live = [
              {{ id: "m1", role: "user", createdAt: new Date(1), content: [{{ type: "text", text: "hi" }}] }},
              {{ id: "m2", role: "assistant", createdAt: new Date(2), content: [{{ type: "text", text: "yo" }}] }},
            ];
            setActiveBranchReader(() => live.slice());
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const pending = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));
            if ({str(empties).lower()}) live = [];
            release();
            await pending;

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1
    assert (out["contextUsage"] or {}).get(
        "totalTokens"
    ) == expected_total, "a total for a branch that has since been emptied must not reach the bar"


@pytest.mark.parametrize(
    ("saved", "expect_counts", "expect_total", "expect_completion"),
    [
        (
            "{ totalTokens: 900, promptTokens: 700, completionTokens: 200, "
            'modelId: "unsloth/gguf-model" }',
            0,
            900,
            200,
        ),
        (
            '{ totalTokens: 900, promptTokens: 700, completionTokens: 200, modelId: "other" }',
            1,
            12,
            0,
        ),
        ("null", 1, 12, 0),
    ],
    ids = ["saved_matches_the_model", "saved_is_another_model", "nothing_saved"],
)
def test_history_hydration_keeps_saved_usage_it_restored(
    saved, expect_counts, expect_total, expect_completion
):
    """The loader restores the last completion's own usage, exact and split into prompt and
    completion. refreshContextUsage does not stand down for a value that was already there,
    so recounting on top would publish an estimate with completionTokens 0 over it."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ hydrateThreadUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{
              activeThreadId: "thread-a",
              contextUsage: null,
              contextUsageByThreadId: {{}},
            }});
            await hydrateThreadUsage({{ remoteId: "thread-a", savedUsage: {saved} }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == expect_counts, (
        "usable saved usage must not be recounted over"
        if expect_counts == 0
        else "without usable saved usage the branch still has to be priced"
    )
    usage = out["contextUsage"] or {}
    assert usage.get("totalTokens") == expect_total
    assert (
        usage.get("completionTokens") == expect_completion
    ), "the completion half of an exact total must survive hydration"


# 100 tokens at the estimator's 4 characters a token.
STORED_TURN = (
    '[{ id: "u1", parentId: null, role: "user", createdAt: 1, '
    'content: [{ type: "text", text: "x".repeat(400) }] }]'
)


@pytest.mark.parametrize(
    ("saved", "expect_shown"),
    [
        ("null", {"totalTokens": 100, "completionTokens": 0, "estimated": True}),
        (
            '{ totalTokens: 900, promptTokens: 700, completionTokens: 200, modelId: "other" }',
            {"totalTokens": 100, "completionTokens": 0, "estimated": True},
        ),
        (
            "{ totalTokens: 900, promptTokens: 700, completionTokens: 200, "
            'modelId: "unsloth/gguf-model" }',
            {"totalTokens": 900, "completionTokens": 200, "estimated": None},
        ),
    ],
    ids = ["nothing_saved", "saved_is_another_model", "saved_matches_the_model"],
)
def test_history_hydration_shows_an_estimate_until_the_recount_lands(saved, expect_shown):
    """The estimate uses the loader's own msgs, which the harness must bind or every history case throws."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ hydrateThreadUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{
              activeThreadId: "thread-a",
              contextUsage: null,
              contextUsageByThreadId: {{}},
            }});
            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            await hydrateThreadUsage({{
              remoteId: "thread-a",
              savedUsage: {saved},
              messages: {STORED_TURN},
            }});
            const shown = snapshot().contextUsage;
            release();
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{
              shown,
              counts: world.countedMessages.length,
              settled: snapshot().contextUsage,
            }}));
            """
        )
    )
    shown = out["shown"] or {}
    assert {key: shown.get(key) for key in expect_shown} == expect_shown, (
        "the bar must show the restored usage, or the estimate when none is usable, "
        "before the recount answers"
    )
    if expect_shown["estimated"]:
        assert out["counts"] == 1, "an estimate is a placeholder; the recount still has to run"
        settled = out["settled"] or {}
        assert settled.get("totalTokens") == 12 and not settled.get(
            "estimated"
        ), "the recount's exact count must replace the estimate"
    else:
        assert out["counts"] == 0


_SIMPLE_BINDING = re.compile(r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)")
_PATTERN_START = re.compile(r"\b(?:const|let)\s*([{\[])")


def _declared_names(code: str) -> set[str]:
    """Names bound by const or let, destructured ones included; nested patterns only over-collect."""
    names = set(_SIMPLE_BINDING.findall(code))
    for start in _PATTERN_START.finditer(code):
        depth, end = 0, start.start(1)
        for end in range(start.start(1), len(code)):
            if code[end] in "{[":
                depth += 1
            elif code[end] in "}]":
                depth -= 1
            if depth == 0:
                break
        for part in re.split(r"[,{}\[\]]", code[start.start(1) : end + 1]):
            part = part.split("=", 1)[0].strip().removeprefix("...")
            if ":" in part:
                part = part.split(":", 1)[1].strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", part):
                names.add(part)
    return names


def test_declared_names_reads_destructuring():
    code = "const { remoteId, threadId: tid, mode = 1, ...rest } = x; let [first, , third] = y; const a = 1;"
    assert _declared_names(code) == {"remoteId", "tid", "mode", "rest", "first", "third", "a"}


def test_the_harness_binds_every_loader_local_the_history_restore_reads() -> None:
    """Locals the sliced restore block reads from above its slice must be declared by hydrateThreadUsage."""
    provider = read(PROVIDER)
    restore = _history_usage_restore()
    start = provider.index(restore)
    loader = provider.rindex("async load() {", 0, start)
    above = provider[loader:start]
    declared_above = _declared_names(above)
    assert (
        {"msgs", "branch", "store"} <= declared_above
    ), "could not read the loader's locals above the restore; this guard would check nothing"
    # Code only: the block's comments name words that are locals elsewhere.
    code = re.sub(r"//[^\n]*", "", restore)
    declared_in_slice = _declared_names(code)
    read_from_above = sorted(
        name
        for name in declared_above - declared_in_slice
        if re.search(rf"(?<![\w$.]){re.escape(name)}\b", code)
    )
    # The block's own inputs change (#13109 moved the estimate from `msgs` to the selected
    # `branch`), so pin only what proves the scan works: the destructured `remoteId`, and at least
    # one plain `const` from above the slice.
    assert "remoteId" in read_from_above and set(read_from_above) - {
        "remoteId"
    }, f"the guard no longer sees the restore read loader locals, only {read_from_above}"
    bound = _declared_names(HARNESS_HISTORY)
    missing = [name for name in read_from_above if name not in bound]
    assert not missing, (
        f"the history restore reads the loader locals {missing}, which hydrateThreadUsage does "
        "not declare: the replay would throw a ReferenceError. Bind them from `props`."
    )


def test_deep_research_recounts_before_the_model_decides():
    """Deep Research does not guarantee a research run, so its request is priced like any other send."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ refreshContextUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ activeThreadId: "thread-a", contextUsage: null, deepResearchEnabled: true }});
            await refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1, "the model-decision turn must be priced before it can hand off"
    assert out["contextUsage"] is not None


def test_an_image_branch_is_declined_before_it_is_sent():
    """The endpoint always 503s on images: /apply-template swaps each one for a short marker.
    Declining client-side keeps the base64 out of the branch hash and out of a request body that
    can run to megabytes, both of which are synchronous work on the UI thread."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage, seed, setActiveBranchReader, snapshot, world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            const live = [
              {{ id: "m1", role: "user", createdAt: new Date(1), content: [
                {{ type: "text", text: "what is this" }},
                {{ type: "image", image: "data:image/png;base64,AAAA" }},
              ] }},
            ];
            setActiveBranchReader(() => live.slice());
            seed({{ activeThreadId: "thread-a", contextUsage: null }});
            await refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 0, "an image branch must never reach the endpoint"
    assert out["contextUsage"] is None


def test_a_second_trigger_does_not_duplicate_an_in_flight_count():
    """Dedupe must skip the second request itself; discarding only its result still does the work."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ refreshContextUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ activeThreadId: "thread-a", contextUsage: null }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const first = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));
            // The second trigger, while the first is still on the wire.
            const second = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            release();
            await Promise.all([first, second]);
            // Settle: the deferred trigger is replayed off the first count's finally, so a stray
            // replay would land after the awaits above and this is what would see it.
            await new Promise((resolve) => setTimeout(resolve, 40));

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1, "the second trigger must not put a second count on the wire"
    assert (out["contextUsage"] or {}).get("totalTokens") == 12, "and the first must still publish"


@pytest.mark.parametrize(
    ("pane", "expect_counts"),
    [
        ('{ modelType: "base" }', 1),
        ('{ modelType: "base", pairId: "pair-1" }', 0),
        ('{ modelType: "finetuned", pairId: "pair-1" }', 0),
    ],
    ids = ["primary_pane", "compare_pane", "compare_finetuned_pane"],
)
def test_only_the_primary_pane_recounts_on_history_load(pane, expect_counts):
    """refreshContextUsage drops a total whose thread is not activeThreadId, and a compare pane
    deliberately never writes that. Counting there rebuilds the branch from storage and pays for
    /apply-template and /tokenize to produce a number that cannot be shown."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ hydrateThreadUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});
            await hydrateThreadUsage({{ remoteId: "thread-a", savedUsage: null, ...{pane} }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{ counts: world.countedMessages.length }}));
            """
        )
    )
    assert out["counts"] == expect_counts, (
        "the primary pane still prices its branch"
        if expect_counts
        else "a compare pane must not spend a count it can never display"
    )


def test_a_trigger_skipped_behind_an_in_flight_count_is_replayed():
    """A trigger skipped behind an in-flight count must be deferred and replayed, or the bar stays blank."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage, seed, setActiveBranchReader, snapshot, world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            const live = [
              {{ id: "m1", role: "user", createdAt: new Date(1), content: [{{ type: "text", text: "hi" }}] }},
            ];
            setActiveBranchReader(() => live.slice());
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const first = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));

            // A turn lands, so the first count is now pricing a branch that moved: it will be
            // dropped at publish. The stopped run's retry arrives while it is still in flight.
            live.push({{
              id: "m2", role: "assistant", createdAt: new Date(2),
              content: [{{ type: "text", text: "yo" }}],
            }});
            const second = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});

            release();
            await Promise.all([first, second]);
            await new Promise((resolve) => setTimeout(resolve, 40));

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 2, (
        "the skipped trigger must be replayed once the stale count settles, or it is not "
        "deferred but lost"
    )
    # 62, not 12: the replay prices the branch as it moved.
    assert (
        (out["contextUsage"] or {}).get("totalTokens") == 62
    ), "and the replay must publish the current branch, which is why it is deferred not dropped"


def test_a_new_chat_recount_is_retried_after_a_background_run_ends():
    """Unpersisted New Chat has no activeThreadId, so the effect must watch the run to retry the count."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ runningByThreadId: {{ "thread-a": true }} }});
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            const during = {{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }};

            // The background run lands. In the app that store write re-renders the component.
            seed({{ runningByThreadId: {{}} }});
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            console.log(JSON.stringify({{
              during,
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert (
        out["during"]["counts"] == 0
    ), "a New Chat must not count while the outgoing conversation is still generating"
    assert out["during"]["contextUsage"] is None
    assert out["counts"] == 1, (
        "the run ending must re-fire this effect: nothing else can, so without it the count is "
        "not deferred but lost"
    )
    assert (out["contextUsage"] or {}).get("totalTokens") == 12


@pytest.mark.parametrize(
    ("first_run_starts", "expected_total"),
    [(True, None), (False, 12)],
    ids = ["first_turn_sent_mid_count", "still_empty"],
)
def test_a_new_chat_count_is_dropped_once_its_first_run_starts(first_run_starts, expected_total):
    """A New Chat count captures a null thread id, so it has no branch to compare. A first turn
    files its run under "__default", the only witness that the bare-template total is now stale."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ refreshContextUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ activeThreadId: null, contextUsage: null, contextUsageByThreadId: {{}} }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const pending = refreshContextUsage({{ afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));
            if ({str(first_run_starts).lower()}) {{
              seed({{ runningByThreadId: {{ __default: true }} }});
            }}
            release();
            await pending;

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1
    assert (out["contextUsage"] or {}).get(
        "totalTokens"
    ) == expected_total, "a bare-template total must not land on a New Chat that already has a turn"


def test_adopting_the_resident_gguf_reprices_the_open_thread():
    """Switching back to a local model that never left memory takes loadModel's already-resident
    branch, which returns before the post-load recount. setCheckpoint blanks the bar and a mounted
    thread does not rerun its history loader, so this branch must recount or the bar stays empty."""
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { adoptResidentModel, seed, snapshot, world } from "./harness.ts";
            world.storedMessages["thread-a"] = [
              { id: "m1", role: "user", createdAt: 1, content: [{ type: "text", text: "hi" }], metadata: {} },
              { id: "m2", role: "assistant", createdAt: 2, content: [{ type: "text", text: "yo" }], metadata: {} },
            ];
            // On an external provider, showing the usage that provider's last turn wrote.
            seed({
              params: { checkpoint: "openai:gpt-4o", systemPrompt: "", systemVariables: "" },
              loadedContextLength: null,
              activeThreadId: "thread-a",
              contextUsage: { promptTokens: 900, completionTokens: 30, totalTokens: 930, cachedTokens: 0 },
            });

            await adoptResidentModel({
              modelId: "unsloth/gguf-model",
              residentStatus: {
                active_model: "unsloth/gguf-model",
                gguf_variant: null,
                is_gguf: true,
                context_length: 8192,
              },
            });
            await new Promise((resolve) => setTimeout(resolve, 30));

            const after = snapshot();
            console.log(JSON.stringify({
              counts: world.countedMessages.length,
              contextUsage: after.contextUsage,
              cached: after.contextUsageByThreadId["thread-a"] ?? null,
            }));
            """
        )
    )
    assert out["counts"] == 1
    assert (out["contextUsage"] or {}).get("totalTokens") == 62, (
        "adopting the resident GGUF must reprice the open thread: setCheckpoint has "
        "already blanked the external provider's usage"
    )
    assert (out["cached"] or {}).get("totalTokens") == 62


def test_status_poll_adoption_reprices_when_a_local_checkpoint_is_already_selected():
    """#10337: an API-loaded GGUF already selected in Studio left the bar blank until the next reply."""
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { adoptServerStatus, seed, snapshot, world } from "./harness.ts";
            world.storedMessages["thread-a"] = [
              { id: "m1", role: "user", createdAt: 1, content: [{ type: "text", text: "hi" }], metadata: {} },
              { id: "m2", role: "assistant", createdAt: 2, content: [{ type: "text", text: "yo" }], metadata: {} },
            ];
            seed({
              params: { checkpoint: "unsloth/gguf-model", systemPrompt: "", systemVariables: "" },
              loadedContextLength: 8192,
              activeThreadId: "thread-a",
              contextUsage: null,
              contextUsageByThreadId: {},
            });

            await adoptServerStatus({
              active_model: "unsloth/gguf-model",
              gguf_variant: null,
              is_gguf: true,
              context_length: 8192,
            });
            await new Promise((resolve) => setTimeout(resolve, 30));

            const after = snapshot();
            console.log(JSON.stringify({
              counts: world.countedMessages.length,
              contextUsage: after.contextUsage,
              cached: after.contextUsageByThreadId["thread-a"] ?? null,
            }));
            """
        )
    )
    assert out["counts"] == 1
    assert (out["contextUsage"] or {}).get("totalTokens") == 62, (
        "adopting a resident GGUF through the status poll must reprice the open "
        "thread even when Studio already had that checkpoint selected"
    )
    assert (out["cached"] or {}).get("totalTokens") == 62


# setCheckpoint emptied contextUsageByThreadId and the mounted loader never reruns.
REVISIT_AFTER_A_MODEL_SWITCH = """
    seed({
      activeThreadId: "thread-b",
      contextUsage: { promptTokens: 700, completionTokens: 20, totalTokens: 720, cachedTokens: 0 },
      contextUsageByThreadId: {
        "thread-a": { promptTokens: 500, completionTokens: 10, totalTokens: 510, cachedTokens: 0 },
        "thread-b": { promptTokens: 700, completionTokens: 20, totalTokens: 720, cachedTokens: 0 },
      },
    });
    renderThreadContextUsageRecount();
    const beforeSwitch = world.countedMessages.length;

    // The user loads a different GGUF, then clicks back into thread-a in the sidebar.
    useChatRuntimeStore.getState().setCheckpoint("unsloth/other-gguf");
    useChatRuntimeStore.getState().setActiveThreadId("thread-a");
    renderThreadContextUsageRecount();
"""

# The loader's recount runs before status answers, which lands before activeThreadId.
DEEP_LINK_HYDRATING_AFTER_THE_LOADER = """
    renderThreadContextUsageRecount();
    await refreshContextUsage({ threadId: "thread-a" });
    const beforeSwitch = world.countedMessages.length;

    // /api/inference/status answers while the thread is still not active.
    seed({
      params: { checkpoint: "unsloth/gguf-model", systemPrompt: "", systemVariables: "" },
      loadedContextLength: 8192,
      modelLoading: false,
    });
    renderThreadContextUsageRecount();

    // ThreadAutoSwitch finally writes the active thread.
    useChatRuntimeStore.getState().setActiveThreadId("thread-a");
    renderThreadContextUsageRecount();
"""


@pytest.mark.parametrize(
    ("seed_script", "scenario"),
    [
        pytest.param(LOADED_MODEL, REVISIT_AFTER_A_MODEL_SWITCH, id = "revisit_a_cached_thread"),
        pytest.param("", DEEP_LINK_HYDRATING_AFTER_THE_LOADER, id = "deep_link_hydrates_late"),
    ],
)
def test_a_thread_becoming_active_with_a_blank_bar_is_repriced(seed_script, scenario):
    """#7450 again, from the two orderings the load-time and history-load recounts miss. Both end
    with a thread the bar points at, a known model and window, and no usage, so the recount has to
    run off the thread becoming active rather than off either independently timed callback."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage,
              renderThreadContextUsageRecount,
              seed,
              snapshot,
              useChatRuntimeStore,
              world,
            }} from "./harness.ts";
            {seed_script}
            {TWO_STORED_TURNS}
            {scenario}
            await new Promise((resolve) => setTimeout(resolve, 30));

            const after = snapshot();
            console.log(JSON.stringify({{
              beforeSwitch,
              counts: world.countedMessages.length,
              sent: world.countedMessages.at(-1) ?? [],
              contextUsage: after.contextUsage,
              cached: after.contextUsageByThreadId["thread-a"] ?? null,
            }}));
            """
        )
    )
    assert out["beforeSwitch"] == 0, "nothing to reprice until the thread is the active one"
    assert (out["contextUsage"] or {}).get("totalTokens") == 62, (
        "a thread the bar points at with no cached usage stays blank until the next "
        "completion unless becoming active reprices it"
    )
    assert (out["cached"] or {}).get("totalTokens") == 62
    assert out["counts"] == 1
    assert len(out["sent"]) == 2, "the thread's stored branch must be priced"


@pytest.mark.parametrize(
    ("picked", "saved_model", "expected"),
    [
        pytest.param(
            "external::openai::gpt-x",
            None,
            {"totalTokens": 6, "estimated": True},
            id = "cloud_model_estimates",
        ),
        pytest.param("", None, {"totalTokens": 6, "estimated": True}, id = "no_model_estimates"),
        pytest.param(
            "external::openai::gpt-x",
            "external::openai::gpt-x",
            {"totalTokens": 300, "estimated": None},
            id = "cloud_model_restores_its_own_saved_usage",
        ),
        pytest.param(
            "external::openai::gpt-x",
            "unsloth/gguf-model",
            {"totalTokens": 6, "estimated": True},
            id = "another_models_saved_usage_is_not_restored",
        ),
    ],
)
def test_a_mounted_thread_revisited_without_a_countable_model_is_refilled(
    picked, saved_model, expected
):
    """#10337: picking a cloud model (or none) wipes every thread's usage, and a chat already opened
    this session never reruns the history loader, so switching back to it showed no bar at all."""
    saved = (
        f'{{ promptTokens: 280, completionTokens: 20, totalTokens: 300, cachedTokens: 0, modelId: "{saved_model}" }}'
        if saved_model
        else "undefined"
    )
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world }} from "./harness.ts";
            {LOADED_MODEL}
            world.storedMessages["thread-a"] = [
              {{ id: "m1", role: "user", createdAt: 1, content: [{{ type: "text", text: "hello there" }}], metadata: {{}} }},
              {{ id: "m2", role: "assistant", createdAt: 2, content: [{{ type: "text", text: "general kenobi" }}],
                metadata: {{ contextUsage: {saved} }} }},
            ];
            seed({{
              activeThreadId: "thread-b",
              contextUsage: {{ promptTokens: 700, completionTokens: 20, totalTokens: 720, cachedTokens: 0 }},
              contextUsageByThreadId: {{
                "thread-a": {{ promptTokens: 500, completionTokens: 10, totalTokens: 510, cachedTokens: 0 }},
                "thread-b": {{ promptTokens: 700, completionTokens: 20, totalTokens: 720, cachedTokens: 0 }},
              }},
            }});
            renderThreadContextUsageRecount();

            // Both chats stay mounted; the user picks another model, then clicks back into thread-a.
            useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});
            if (!{json.dumps(picked)}) seed({{ loadedContextLength: null }});
            renderThreadContextUsageRecount();
            useChatRuntimeStore.getState().setActiveThreadId("thread-a");
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));

            const after = snapshot();
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: after.contextUsage,
              cached: after.contextUsageByThreadId["thread-a"] ?? null,
            }}));
            """
        )
    )
    assert out["counts"] == 0, "nothing can be counted exactly without a local model"
    shown = out["contextUsage"] or {}
    assert (
        shown.get("totalTokens") == expected["totalTokens"]
    ), "a revisited chat must show its stored usage or an estimate, not a blank bar"
    assert shown.get("estimated") == expected["estimated"]
    assert (out["cached"] or {}).get("totalTokens") == expected["totalTokens"]


@pytest.mark.parametrize(
    ("run_usage", "expected"),
    [
        ("null", {"totalTokens": 13, "estimated": True}),
        (
            "{ promptTokens: 40, completionTokens: 9, totalTokens: 49, cachedTokens: 0 }",
            {"totalTokens": 49, "estimated": None},
        ),
    ],
    ids = ["run_ends_without_usage", "run_reports_usage"],
)
def test_a_cloud_estimate_is_repriced_after_a_run_but_exact_usage_stays(run_usage, expected):
    """A cancelled or failed cloud run, or a provider that omits usage, leaves the pre-run estimate
    on the bar; the run ending must re-estimate the new turn, while a reported count is kept."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world }} from "./harness.ts";
            {LOADED_MODEL}
            world.storedMessages["thread-a"] = [
              {{ id: "m1", role: "user", createdAt: 1, content: [{{ type: "text", text: "hello there" }}], metadata: {{}} }},
              {{ id: "m2", role: "assistant", createdAt: 2, content: [{{ type: "text", text: "general kenobi" }}], metadata: {{}} }},
            ];
            seed({{ activeThreadId: "thread-a" }});
            useChatRuntimeStore.getState().setCheckpoint("external::openai::gpt-x");
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));
            const before = snapshot().contextUsage;

            seed({{ runningByThreadId: {{ "thread-a": true }} }});
            renderThreadContextUsageRecount();
            world.storedMessages["thread-a"].push(
              {{ id: "m3", role: "user", createdAt: 3, content: [{{ type: "text", text: "and what about the moon then" }}], metadata: {{}} }},
            );
            const reported = {run_usage};
            if (reported) useChatRuntimeStore.getState().setContextUsage(reported);
            seed({{ runningByThreadId: {{}} }});
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{ before, after: snapshot().contextUsage }}));
            """
        )
    )
    assert (out["before"] or {}).get("totalTokens") == 6
    after = out["after"] or {}
    assert after.get("totalTokens") == expected["totalTokens"]
    assert after.get("estimated") == expected["estimated"]


@pytest.mark.parametrize("thread_id", ["thread-a", "__LOCALID_saved"])
def test_the_cloud_refill_prices_the_mounted_branch_without_reading_storage(thread_id):
    """The refill runs only for a mounted chat, whose runtime already holds the visible branch. A
    storage read would 404 for a just-sent chat (Studio UI CI, IME smoke), and saved chats can carry
    a runtime-minted __LOCALID_ id, so the id shape must not decide anything."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world }} from "./harness.ts";
            let reads = 0;
            const records = [
              {{ id: "m1", role: "user", createdAt: 1, content: [{{ type: "text", text: "hello there" }}], metadata: {{}} }},
              {{ id: "m2", role: "assistant", createdAt: 2, content: [{{ type: "text", text: "general kenobi" }}], metadata: {{}} }},
            ];
            world.liveMessages = {{ "{thread_id}": records.map((r) => ({{ ...r, createdAt: new Date(r.createdAt), metadata: {{ custom: {{}} }} }})) }};
            world.storedMessages = new Proxy({{}}, {{ get() {{ reads += 1; return []; }} }});
            seed({{ activeThreadId: "{thread_id}" }});
            useChatRuntimeStore.getState().setCheckpoint("external::openai::gpt-x");
            renderThreadContextUsageRecount();
            console.log(JSON.stringify({{ reads, contextUsage: snapshot().contextUsage }}));
            """
        )
    )
    assert out["reads"] == 0, "the mounted branch is in memory; nothing should hit storage"
    assert (out["contextUsage"] or {}).get("totalTokens") == 6


IMAGE_ONLY = '[{ type: "image", image: "data:image/png;base64,iVBORw0KGgo=" }]'


@pytest.mark.parametrize(
    ("parts", "saved", "gguf", "expected"),
    [
        # The recount prices a text-only branch exactly; the estimate must stay out of its way.
        (
            '[{ type: "text", text: "describe this please" }]',
            "undefined",
            True,
            {"total": 62, "estimated": None, "counts": 1},
        ),
        # It declines an image branch, which left the bar blank after a local model switch.
        (
            '[{ type: "text", text: "describe this please" }, { type: "image", image: "data:image/png;base64,iVBORw0KGgo=" }]',
            "undefined",
            True,
            {"total": 9, "estimated": True, "counts": 0},
        ),
        # That model's own saved count is restored, as the history loader would ...
        (
            IMAGE_ONLY,
            '{ promptTokens: 700, completionTokens: 9, totalTokens: 709, cachedTokens: 0, modelId: "unsloth/other-gguf" }',
            True,
            {"total": 709, "estimated": None, "counts": 0},
        ),
        # ... unless it no longer fits a llama.cpp window ...
        (
            IMAGE_ONLY,
            '{ promptTokens: 9000, completionTokens: 9, totalTokens: 9009, cachedTokens: 0, modelId: "unsloth/other-gguf" }',
            True,
            {"total": 4, "estimated": True, "counts": 0},
        ),
        # ... which MLX runs past, so there the count stands ...
        (
            IMAGE_ONLY,
            '{ promptTokens: 9000, completionTokens: 9, totalTokens: 9009, cachedTokens: 0, modelId: "unsloth/other-gguf" }',
            False,
            {"total": 9009, "estimated": None, "counts": 0},
        ),
        # ... and legacy usage with no modelId is trusted under a known local window.
        (
            IMAGE_ONLY,
            "{ promptTokens: 300, completionTokens: 9, totalTokens: 309, cachedTokens: 0 }",
            True,
            {"total": 309, "estimated": None, "counts": 0},
        ),
    ],
    ids = [
        "text_only_is_counted",
        "image_is_estimated",
        "image_restores_its_saved_count",
        "image_saved_count_past_gguf_window",
        "image_saved_count_past_mlx_window",
        "image_restores_legacy_usage",
    ],
)
def test_a_media_chat_keeps_a_value_after_a_local_model_switch(parts, saved, gguf, expected):
    """The recount refuses images, audio and video, so a chat carrying one showed a blank bar once
    another local model loaded under it; the mounted branch fills it by the history loader's rules."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ loadedIsGguf: {"true" if gguf else "false"} }});
            world.storedMessages["thread-a"] = [
              {{ id: "m1", role: "user", createdAt: 1, content: {parts}, metadata: {{}} }},
              {{ id: "m2", role: "assistant", createdAt: 2, content: [{{ type: "text", text: "a cat on a sofa" }}],
                metadata: {{ contextUsage: {saved} }} }},
            ];
            seed({{ activeThreadId: "thread-a", contextUsage: {{ promptTokens: 900, completionTokens: 9, totalTokens: 909, cachedTokens: 0 }} }});
            renderThreadContextUsageRecount();
            useChatRuntimeStore.getState().setCheckpoint("unsloth/other-gguf");
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));
            const after = snapshot();
            console.log(JSON.stringify({{ counts: world.countedMessages.length, usage: after.contextUsage }}));
            """
        )
    )
    usage = out["usage"] or {}
    assert out["counts"] == expected["counts"]
    assert (
        usage.get("totalTokens") == expected["total"]
    ), "a media chat must not be left with a blank bar"
    assert usage.get("estimated") == expected["estimated"]


@pytest.mark.parametrize(
    ("picked", "tail", "saved", "expected"),
    [
        # A resident GGUF stays loaded under a cloud pick; its window must not cap the cloud's count.
        (
            "external::openai::gpt-x",
            "",
            '{ promptTokens: 19000, completionTokens: 9, totalTokens: 20009, cachedTokens: 0, modelId: "external::openai::gpt-x" }',
            {"total": 20009, "estimated": None},
        ),
        # A media turn whose run never answered follows the counted reply: that count is stale.
        (
            "unsloth/other-gguf",
            ', { id: "m3", role: "user", createdAt: 3, content: [{ type: "image", image: "data:image/png;base64,iVBORw0KGgo=" }], metadata: {} }',
            '{ promptTokens: 700, completionTokens: 9, totalTokens: 709, cachedTokens: 0, modelId: "unsloth/other-gguf" }',
            {"total": 709, "estimated": True},
        ),
    ],
    ids = ["cloud_count_past_a_resident_gguf_window", "unanswered_media_turn_after_the_count"],
)
def test_the_refill_restores_only_a_count_that_describes_the_branch(picked, tail, saved, expected):
    """A saved count is restored only when it is the branch tip's and the SELECTED model's window
    (not a resident GGUF's under a cloud pick) admits it; otherwise the branch is estimated."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{ loadedIsGguf: true }});
            world.storedMessages["thread-a"] = [
              {{ id: "m1", role: "user", createdAt: 1, content: [{{ type: "text", text: "describe this please" }}], metadata: {{}} }},
              {{ id: "m2", role: "assistant", createdAt: 2, content: [{{ type: "text", text: "a cat on a sofa" }}],
                metadata: {{ contextUsage: {saved} }} }}{tail}
            ];
            seed({{ activeThreadId: "thread-a" }});
            renderThreadContextUsageRecount();
            useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({{ usage: snapshot().contextUsage }}));
            """
        )
    )
    usage = out["usage"] or {}
    assert usage.get("totalTokens") == expected["total"]
    assert usage.get("estimated") == expected["estimated"]


def test_the_cloud_refill_never_prices_another_mounted_thread():
    """New Chat leaves the outgoing conversation mounted until the switch settles, so the runtime's
    main thread can lag the store; pricing it then would put another chat's count on the bar."""
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
            world.storedMessages["thread-a"] = [
              { id: "m1", role: "user", createdAt: 1, content: [{ type: "text", text: "hello there" }], metadata: {} },
            ];
            world.mainThreadId = "thread-b";
            seed({ activeThreadId: "thread-a" });
            useChatRuntimeStore.getState().setCheckpoint("external::openai::gpt-x");
            renderThreadContextUsageRecount();
            console.log(JSON.stringify({ contextUsage: snapshot().contextUsage }));
            """
        )
    )
    assert out["contextUsage"] is None


@pytest.mark.parametrize("interruption", ["run_active", "load_still_cancelling"])
def test_a_cloud_refill_waits_for_a_run_or_load_to_settle(interruption):
    """Nothing is published while a run or a model load is in flight (a cancelled local load clears
    usage again after the external pick); both settling must re-fire the refill on the new branch."""
    if interruption == "run_active":
        interrupt = 'seed({ runningByThreadId: { "thread-a": true } });'
        settle = """
            world.storedMessages["thread-a"].push(
              { id: "m3", role: "user", createdAt: 3, content: [{ type: "text", text: "and what about the moon then" }], metadata: {} },
            );
            seed({ runningByThreadId: {} });
        """
    else:
        interrupt = "seed({ modelLoading: true });"
        settle = "seed({ modelLoading: false, contextUsage: null, contextUsageByThreadId: {} });"
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world }} from "./harness.ts";
            {LOADED_MODEL}
            world.storedMessages["thread-a"] = [
              {{ id: "m1", role: "user", createdAt: 1, content: [{{ type: "text", text: "hello there" }}], metadata: {{}} }},
              {{ id: "m2", role: "assistant", createdAt: 2, content: [{{ type: "text", text: "general kenobi" }}], metadata: {{}} }},
            ];
            seed({{ activeThreadId: "thread-a" }});
            useChatRuntimeStore.getState().setCheckpoint("external::openai::gpt-x");
            {interrupt}
            renderThreadContextUsageRecount();
            const during = snapshot().contextUsage;

            {settle}
            renderThreadContextUsageRecount();
            console.log(JSON.stringify({{ during, after: snapshot().contextUsage }}));
            """
        )
    )
    assert out["during"] is None, "nothing may be published while a run or load is in flight"
    expected = 13 if interruption == "run_active" else 6
    assert (
        (out["after"] or {}).get("totalTokens") == expected
    ), "the refill must run again once the run or load settles, against the current branch"


@pytest.mark.parametrize(
    ("mount", "expected_total"),
    [
        (
            'live = [...stored, { id: "m3", role: "user", createdAt: new Date(3),'
            ' content: [{ type: "text", text: "and another" }] }];',
            None,
        ),
        ("live = stored.slice();", 62),
        ("", 62),
    ],
    ids = ["mounted_with_a_new_turn", "mounted_unchanged", "still_unmounted"],
)
def test_a_stored_history_count_is_dropped_once_the_runtime_contradicts_it(mount, expected_total):
    """A recount that falls back to storage runs before the thread is mounted, so it has no live
    branch to sign. Ids survive both shapes: a different mounted tail means the total is stale."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{
              refreshContextUsage,
              seed,
              setActiveBranchReader,
              snapshot,
              world,
            }} from "./harness.ts";
            {LOADED_MODEL}
            {TWO_STORED_TURNS}
            const stored = [
              {{ id: "m1", role: "user", createdAt: new Date(1), content: [{{ type: "text", text: "hi" }}] }},
              {{ id: "m2", role: "assistant", createdAt: new Date(2), content: [{{ type: "text", text: "yo" }}] }},
            ];
            let live = null;
            setActiveBranchReader(() => live);
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});

            let release;
            world.countGate = new Promise((resolve) => {{ release = resolve; }});
            const pending = refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            await new Promise((resolve) => setTimeout(resolve, 20));
            {mount}
            release();
            await pending;

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              countedLength: world.countedMessages[0]?.length ?? 0,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1, "the count is still attempted"
    assert out["countedLength"] == 2, "it priced the two stored turns"
    assert (out["contextUsage"] or {}).get("totalTokens") == expected_total, (
        "a stored total the runtime has since contradicted must not reach the bar"
        if expected_total is None
        else "a stored count with nothing contradicting it must still publish"
    )


@pytest.mark.parametrize(
    ("model_flags", "expected_counts"),
    [
        ("{ isAudio: true, hasAudioInput: false }", 0),
        ("{ isAudio: true, hasAudioInput: true }", 1),
        ("{ isAudio: false, hasAudioInput: false }", 1),
    ],
    ids = ["output_only_audio", "audio_input_model", "plain_gguf"],
)
def test_an_output_only_audio_gguf_is_never_recounted(model_flags, expected_counts):
    """A TTS GGUF sends through /audio/generate, which answers with no usage. A chat-template total
    over the thread would park a number on the bar that describes nothing and nothing corrects."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ refreshContextUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{
              activeThreadId: "thread-a",
              contextUsage: null,
              contextUsageByThreadId: {{}},
              models: [{{ id: "unsloth/gguf-model", ...{model_flags} }}],
            }});
            await refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == expected_counts, (
        "an output-only audio model must not be counted at all"
        if expected_counts == 0
        else "a model that answers with a chat completion must still be counted"
    )
    if expected_counts == 0:
        assert out["contextUsage"] is None, "the bar stays blank, as it did before the recount"


@pytest.mark.parametrize(
    ("local_runs", "expected_counts"),
    [
        # The count would share the local llama-server with generation.
        ('{ "thread-a": true }', 0),
        ('{ "thread-b": true }', 0),
        ("{}", 1),
    ],
    ids = ["this_thread_running", "another_thread_running", "nothing_running"],
)
def test_no_count_is_issued_while_anything_is_generating(local_runs, expected_counts):
    """No count while any run is live, local or external, since chat_count_tokens refuses them with 503."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ refreshContextUsage, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            seed({{
              activeThreadId: "thread-a",
              contextUsage: null,
              contextUsageByThreadId: {{}},
              runningByThreadId: {local_runs},
            }});
            await refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == expected_counts, (
        "a count must never be issued while anything is generating"
        if expected_counts == 0
        else "an idle server must still be counted"
    )


def test_the_count_is_retried_once_the_run_finishes():
    """Skipping is only safe because the run finishing re-fires the effect. The recount
    component lists runActive in its dependency array for exactly this reason, so a count
    skipped for being busy is not a count lost."""
    src = read(PROVIDER)
    recount = slice_between(
        src,
        "function ThreadContextUsageRecount(",
        "\nfunction CancelRegistrar(",
    )
    assert "runningByThreadId" in recount, "the effect must observe decoding"
    deps = re.search(r"\}, \[([^\]]*)\]\);", recount, re.S)
    assert deps and "runActive" in deps.group(1), (
        "runActive must be a DEPENDENCY, not just a guard: nothing else in this array "
        "changes when a run ends, so without it a skipped count would never be retried"
    )


_U1_ROW = (
    '{ id: "u1", parentId: null, role: "user", createdAt: 1, '
    'content: [{ type: "text", text: "hello there" }], metadata: {} }'
)

_TOOL_CALL_PARTS = {
    "local": (
        '{ type: "tool-call", toolCallId: "c1", toolName: "web_search", args: {}, '
        'argsText: "{}", result: "found", provenance: { source: "local", round_id: 1 } }, '
    ),
    "hosted": (
        '{ type: "tool-call", toolCallId: "c1", toolName: "web_search", args: {}, '
        'argsText: "{}", result: "found" }, '
    ),
}

_CLOUD_SUM = (
    "{ promptTokens: 322000, completionTokens: 760, totalTokens: 322760, cachedTokens: 0, "
    'modelId: "external::openai::gpt-x" }'
)

_CLOUD_SUM_WITH_CONTEXT = (
    "{ promptTokens: 322000, completionTokens: 760, totalTokens: 322760, contextTokens: 52000, "
    'cachedTokens: 0, modelId: "external::openai::gpt-x" }'
)

_SAVED_300_LOCAL = (
    "{ promptTokens: 250, completionTokens: 50, totalTokens: 300, cachedTokens: 0, "
    'modelId: "unsloth/gguf-model" }'
)

_SAVED_300_CLOUD = (
    "{ promptTokens: 250, completionTokens: 50, totalTokens: 300, cachedTokens: 0, "
    'modelId: "external::openai::gpt-x" }'
)


def _assistant_row(
    id_,
    created_at,
    text,
    saved,
    parent = "u1",
    tool_call = None,
):
    parts = _TOOL_CALL_PARTS[tool_call] if tool_call else ""
    return (
        f'{{ id: "{id_}", parentId: "{parent}", role: "assistant", createdAt: {created_at}, '
        f'content: [{parts}{{ type: "text", text: "{text}" }}], '
        f"metadata: {{ contextUsage: {saved} }} }}"
    )


def _script(template: str, **rows: str) -> str:
    script = textwrap.dedent(template).replace("__LOADED_MODEL__", LOADED_MODEL)
    for name, row in rows.items():
        script = script.replace(f"__{name}__", row)
    return script


_BRANCH_SWITCH = """
    // @ts-nocheck
    import { renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
    __LOADED_MODEL__
    seed({ loadedIsGguf: true });
    __PICK__
    const u1 = __U1__;
    world.storedMessages["thread-a"] = [u1, __A2__];
    seed({ activeThreadId: "thread-a" });
    useChatRuntimeStore.getState().setContextUsage(__FIRST__);
    renderThreadContextUsageRecount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    const before = snapshot().contextUsage;
    const countsBefore = world.countedMessages.length;
    world.storedMessages["thread-a"] = [u1, __A3__];
    renderThreadContextUsageRecount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    console.log(JSON.stringify({
      before,
      after: snapshot().contextUsage,
      counts: world.countedMessages.length - countsBefore,
    }));
"""


@pytest.mark.parametrize(
    ("picked", "first_saved", "switched_saved", "expected"),
    [
        (
            "",
            _SAVED_300_LOCAL,
            "{ promptTokens: 400, completionTokens: 100, totalTokens: 500, cachedTokens: 0, "
            'modelId: "unsloth/gguf-model" }',
            {"counts": 0, "total": 500, "estimated": None},
        ),
        ("", _SAVED_300_LOCAL, "undefined", {"counts": 1, "total": 62, "estimated": None}),
        (
            "external::openai::gpt-x",
            _SAVED_300_CLOUD,
            "{ promptTokens: 400, completionTokens: 100, totalTokens: 500, cachedTokens: 0, "
            'modelId: "external::openai::gpt-x" }',
            {"counts": 0, "total": 500, "estimated": None},
        ),
        (
            "external::openai::gpt-x",
            _SAVED_300_CLOUD,
            "undefined",
            {"counts": 0, "total": 6, "estimated": True},
        ),
    ],
    ids = [
        "local_tip_with_its_own_count",
        "local_tip_is_recounted",
        "cloud_tip_with_its_own_count",
        "cloud_tip_without_a_count_is_estimated_as_reopening_shows",
    ],
)
def test_a_branch_switch_reprices_the_meter_from_the_new_tip(
    picked, first_saved, switched_saved, expected
):
    pick = f"useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});" if picked else ""
    out = _run(
        _script(
            _BRANCH_SWITCH,
            PICK = pick,
            FIRST = first_saved,
            U1 = _U1_ROW,
            A2 = _assistant_row("a2", 2, "general kenobi", first_saved),
            A3 = _assistant_row("a3", 3, "hello to you", switched_saved),
        )
    )
    assert (out["before"] or {}).get("totalTokens") == 300
    after = out["after"] or {}
    assert out["counts"] == expected["counts"]
    assert after.get("totalTokens") == expected["total"]
    assert after.get("estimated") == expected["estimated"]


_SEND_WINDOW = """
    // @ts-nocheck
    import { renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
    __LOADED_MODEL__
    seed({ loadedIsGguf: true });
    __PICK__
    const msg = (id, role, text, extra = {}) => ({
      id, role, createdAt: new Date(), content: [{ type: "text", text }],
      metadata: { custom: extra.usage ? { contextUsage: extra.usage } : {} },
      ...(role === "assistant" ? { status: { type: extra.status ?? "complete" } } : {}),
    });
    const saved = { promptTokens: 250, completionTokens: 50, totalTokens: 300, cachedTokens: 0, modelId: useChatRuntimeStore.getState().params.checkpoint };
    const live = [msg("u1", "user", "hello there"), msg("a2", "assistant", "general kenobi", { usage: saved })];
    world.liveMessages = { "thread-a": live };
    world.storedMessages["thread-a"] = live;
    seed({ activeThreadId: "thread-a" });
    useChatRuntimeStore.getState().setContextUsage(saved);
    renderThreadContextUsageRecount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    const countsBefore = world.countedMessages.length;
    const steps = [];
    const step = async (name) => {
      renderThreadContextUsageRecount();
      await new Promise((resolve) => setTimeout(resolve, 30));
      const usage = snapshot().contextUsage;
      steps.push({ name, counts: world.countedMessages.length - countsBefore, total: usage?.totalTokens ?? null, estimated: usage?.estimated ?? null });
    };
    live.push(msg("u3", "user", "and the moon"), msg("a4", "assistant", "", { status: "running" }));
    await step("running_placeholder");
    seed({ runningByThreadId: { "thread-a": true } });
    await step("running");
    __END__
    console.log(JSON.stringify({ steps }));
"""

_RUN_COMPLETES = """
    const published = { promptTokens: 700, completionTokens: 9, totalTokens: 709, cachedTokens: 0 };
    useChatRuntimeStore.getState().setContextUsage(published);
    seed({ runningByThreadId: {} });
    await step("run_flag_cleared");
    live[3] = msg("a4", "assistant", "is far away", { usage: { ...published, modelId: useChatRuntimeStore.getState().params.checkpoint } });
    await step("tip_settled");
"""

_RUN_CANCELLED = """
    seed({ runningByThreadId: {} });
    await step("run_flag_cleared");
    live[3] = { ...live[3], status: { type: "incomplete", reason: "cancelled" } };
    await step("tip_settled");
"""


@pytest.mark.parametrize(
    "picked", ["", "external::openai::gpt-x"], ids = ["local_model", "cloud_model"]
)
def test_sending_a_message_costs_no_count_and_never_flickers(picked):
    pick = f"useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});" if picked else ""
    out = _run(_script(_SEND_WINDOW, PICK = pick, END = textwrap.dedent(_RUN_COMPLETES)))
    steps = {step["name"]: step for step in out["steps"]}
    for name, step in steps.items():
        assert step["counts"] == 0, name
        assert step["estimated"] is None, name
    for name in ("running_placeholder", "running"):
        assert steps[name]["total"] == 300, name
    assert steps["run_flag_cleared"]["total"] == 709
    assert steps["tip_settled"]["total"] == 709


@pytest.mark.parametrize(
    ("picked", "expected"),
    [
        ("", {"counts": 1, "total": None, "estimated": None}),
        ("external::openai::gpt-x", {"counts": 0, "total": 303, "estimated": True}),
    ],
    ids = ["local_model_recounts", "cloud_model_builds_on_the_last_count"],
)
def test_a_run_cancelled_before_any_output_reprices_only_what_it_can_count(picked, expected):
    pick = f"useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});" if picked else ""
    out = _run(_script(_SEND_WINDOW, PICK = pick, END = textwrap.dedent(_RUN_CANCELLED)))
    steps = {step["name"]: step for step in out["steps"]}
    for name in ("running_placeholder", "running"):
        assert steps[name]["counts"] == 0, name
        assert steps[name]["total"] == 300, name
    settled = steps["tip_settled"]
    assert settled["counts"] == expected["counts"]
    assert settled["estimated"] == expected["estimated"]
    if expected["total"] is not None:
        assert settled["total"] == expected["total"]
    else:
        assert settled["total"] not in (None, 300)


def test_a_runtime_still_showing_another_thread_never_prices_this_one():
    out = _run(
        _script(
            """
            // @ts-nocheck
            import { renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
            __LOADED_MODEL__
            seed({ loadedIsGguf: true });
            const u1 = __U1__;
            world.storedMessages["thread-a"] = [u1, __A2__];
            seed({ activeThreadId: "thread-a" });
            useChatRuntimeStore.getState().setContextUsage(__FIRST__);
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));
            const before = snapshot().contextUsage;
            const countsBefore = world.countedMessages.length;
            world.mainThreadId = "thread-b";
            world.storedMessages["thread-a"] = [u1, __A3__];
            renderThreadContextUsageRecount();
            world.storedMessages["thread-a"] = [];
            renderThreadContextUsageRecount();
            world.mainThreadId = undefined;
            renderThreadContextUsageRecount();
            world.storedMessages["thread-a"] = [u1, __A2__];
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({
              before,
              after: snapshot().contextUsage,
              counts: world.countedMessages.length - countsBefore,
            }));
            """,
            FIRST = _SAVED_300_LOCAL,
            U1 = _U1_ROW,
            A2 = _assistant_row("a2", 2, "general kenobi", _SAVED_300_LOCAL),
            A3 = _assistant_row("a3", 3, "another chat", "undefined"),
        )
    )
    assert (out["before"] or {}).get("totalTokens") == 300
    assert out["counts"] == 0
    assert (out["after"] or {}).get("totalTokens") == 300


@pytest.mark.parametrize(
    ("saved", "tool_call", "expected"),
    [
        (_CLOUD_SUM, "local", {"estimated": True, "total": 7, "context": None}),
        (_CLOUD_SUM_WITH_CONTEXT, "local", {"estimated": None, "total": 322760, "context": 52000}),
        (_CLOUD_SUM, "hosted", {"estimated": None, "total": 322760, "context": None}),
    ],
    ids = ["loop_sum_without_context", "loop_turn_with_context", "provider_hosted_tool_turn"],
)
def test_a_cloud_tool_loop_sum_on_the_tip_is_estimated(saved, tool_call, expected):
    out = _run(
        _script(
            """
            // @ts-nocheck
            import { renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
            __LOADED_MODEL__
            world.storedMessages["thread-a"] = [__U1__, __A2__];
            seed({ activeThreadId: "thread-a" });
            useChatRuntimeStore.getState().setCheckpoint("external::openai::gpt-x");
            renderThreadContextUsageRecount();
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({ usage: snapshot().contextUsage }));
            """,
            U1 = _U1_ROW,
            A2 = _assistant_row("a2", 2, "found it", saved, tool_call = tool_call),
        )
    )
    usage = out["usage"] or {}
    assert usage.get("estimated") == expected["estimated"]
    assert usage.get("totalTokens") == expected["total"]
    assert usage.get("contextTokens") == expected["context"]


@pytest.mark.parametrize(
    ("saved", "tool_call", "expected"),
    [
        (_CLOUD_SUM, "local", {"estimated": True, "total": 104}),
        (_CLOUD_SUM_WITH_CONTEXT, "local", {"estimated": None, "total": 322760}),
        (_CLOUD_SUM, "hosted", {"estimated": None, "total": 322760}),
    ],
    ids = ["loop_sum_without_context", "loop_turn_with_context", "provider_hosted_tool_turn"],
)
def test_history_hydration_estimates_a_cloud_tool_loop_sum(saved, tool_call, expected):
    out = _run(
        _script(
            """
            // @ts-nocheck
            import { hydrateThreadUsage, seed, snapshot, world } from "./harness.ts";
            seed({
              params: { checkpoint: "external::openai::gpt-x", systemPrompt: "", systemVariables: "" },
              loadedContextLength: null,
              loadedIsGguf: false,
              modelLoading: false,
              activeThreadId: "thread-a",
              contextUsage: null,
              contextUsageByThreadId: {},
            });
            const messages = [
              { id: "u1", parentId: null, role: "user", createdAt: 1, content: [{ type: "text", text: "x".repeat(400) }], metadata: {} },
              __A2__,
            ];
            await hydrateThreadUsage({ remoteId: "thread-a", messages });
            await new Promise((resolve) => setTimeout(resolve, 30));
            console.log(JSON.stringify({
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }));
            """,
            A2 = _assistant_row("a2", 2, "found it", saved, tool_call = tool_call),
        )
    )
    assert out["counts"] == 0
    usage = out["contextUsage"] or {}
    assert usage.get("estimated") == expected["estimated"]
    assert usage.get("totalTokens") == expected["total"]


def test_switching_to_a_thread_priced_under_another_model_blanks_the_bar():
    out = _run(
        _script(
            """
            // @ts-nocheck
            import { seed, useChatRuntimeStore } from "./harness.ts";
            __LOADED_MODEL__
            const store = useChatRuntimeStore.getState();
            store.setThreadContextUsage("thread-a", { promptTokens: 1, completionTokens: 1, totalTokens: 2, cachedTokens: 0, modelId: "unsloth/other-gguf" });
            store.setThreadContextUsage("thread-b", { promptTokens: 2, completionTokens: 2, totalTokens: 4, cachedTokens: 0 });
            store.setActiveThreadId("thread-a");
            const a = useChatRuntimeStore.getState().contextUsage;
            store.setActiveThreadId("thread-b");
            const b = useChatRuntimeStore.getState().contextUsage;
            console.log(JSON.stringify({
              a,
              b,
              stamped: useChatRuntimeStore.getState().contextUsageByThreadId["thread-b"].modelId,
            }));
            """
        )
    )
    assert out["a"] is None
    assert (out["b"] or {}).get("totalTokens") == 4
    assert out["stamped"] == "unsloth/gguf-model"


_LIVE_CHANGE = """
    // @ts-nocheck
    import { renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
    __LOADED_MODEL__
    seed({ loadedIsGguf: true });
    __PICK__
    const model = useChatRuntimeStore.getState().params.checkpoint;
    const text = (role, size) => (role === "user" ? "x" : "y").repeat(size);
    const msg = (id, role, at, custom = {}, size = 40) => ({
      id, role, createdAt: new Date(at), content: [{ type: "text", text: text(role, size) }],
      metadata: { custom },
      ...(role === "assistant" ? { status: { type: "complete" } } : {}),
    });
    const saved = (total, messageCount) => ({ promptTokens: total - 10, completionTokens: 10, totalTokens: total, cachedTokens: 0, modelId: model, messageCount });
    const live = [
      msg("u1", "user", 1), msg("a2", "assistant", 2, { contextUsage: saved(300, 2) }),
      msg("u3", "user", 3), msg("a4", "assistant", 4, { contextUsage: saved(500, 4) }),
      msg("u5", "user", 5), msg("a6", "assistant", 6, { contextUsage: saved(700, 6) }),
    ];
    world.liveMessages = { "thread-a": live };
    world.storedMessages["thread-a"] = [];
    seed({ activeThreadId: "thread-a" });
    useChatRuntimeStore.getState().setContextUsage(saved(700, 6));
    renderThreadContextUsageRecount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    const countsBefore = world.countedMessages.length;
    const after = (__AFTER__)(live);
    world.liveMessages["thread-a"] = after;
    world.storedMessages["thread-a"] = after.map((m) => ({ ...m, createdAt: m.createdAt.getTime() }));
    renderThreadContextUsageRecount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    const usage = snapshot().contextUsage;
    console.log(JSON.stringify({
      counts: world.countedMessages.length - countsBefore,
      total: usage?.totalTokens ?? null,
      estimated: usage?.estimated ?? null,
    }));
"""

_LIVE_CHANGES = {
    "in_place_edit_of_the_tip": (
        "(live) => [...live.slice(0, 5), msg('a6', 'assistant', 6, "
        "{ contextUsage: saved(700, 6), revision: 1 }, 400)]"
    ),
    "metadata_only_change": (
        "(live) => [...live.slice(0, 5), { ...live[5], metadata: { custom: "
        "{ ...live[5].metadata.custom, feedback: 'up' } } }]"
    ),
    "delete_the_last_reply": "(live) => live.slice(0, 5)",
    "delete_the_last_turn": "(live) => live.slice(0, 4)",
    "delete_a_middle_turn": "(live) => [live[0], live[1], live[4], live[5]]",
    "delete_every_reply": "(live) => live.slice(0, 1)",
    "delete_everything": "(live) => []",
}


@pytest.mark.parametrize(
    ("change", "picked", "expected"),
    [
        ("in_place_edit_of_the_tip", "", {"counts": 1, "total": 162, "estimated": None}),
        (
            "in_place_edit_of_the_tip",
            "external::openai::gpt-x",
            {"counts": 0, "total": 610, "estimated": True},
        ),
        ("metadata_only_change", "", {"counts": 0, "total": 700, "estimated": None}),
        (
            "metadata_only_change",
            "external::openai::gpt-x",
            {"counts": 0, "total": 700, "estimated": None},
        ),
        ("delete_the_last_reply", "", {"counts": 1, "total": 137, "estimated": None}),
        (
            "delete_the_last_reply",
            "external::openai::gpt-x",
            {"counts": 0, "total": 510, "estimated": True},
        ),
        ("delete_the_last_turn", "", {"counts": 0, "total": 500, "estimated": None}),
        (
            "delete_the_last_turn",
            "external::openai::gpt-x",
            {"counts": 0, "total": 500, "estimated": None},
        ),
        ("delete_a_middle_turn", "", {"counts": 1, "total": 112, "estimated": None}),
        (
            "delete_a_middle_turn",
            "external::openai::gpt-x",
            {"counts": 0, "total": 320, "estimated": True},
        ),
        ("delete_every_reply", "", {"counts": 1, "total": 37, "estimated": None}),
        (
            "delete_every_reply",
            "external::openai::gpt-x",
            {"counts": 0, "total": 10, "estimated": True},
        ),
        ("delete_everything", "", {"counts": 1, "total": 12, "estimated": None}),
        (
            "delete_everything",
            "external::openai::gpt-x",
            {"counts": 0, "total": None, "estimated": None},
        ),
    ],
    ids = lambda value: value if isinstance(value, str) and not value.startswith("{") else None,
)
def test_a_change_to_the_shown_branch_reprices_the_meter(change, picked, expected):
    pick = f"useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});" if picked else ""
    out = _run(_script(_LIVE_CHANGE, PICK = pick, AFTER = _LIVE_CHANGES[change]))
    assert out == expected


_HYDRATE_REVISED = """
    // @ts-nocheck
    import { hydrateThreadUsage, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
    __LOADED_MODEL__
    seed({ loadedIsGguf: true, activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {} });
    __PICK__
    const model = useChatRuntimeStore.getState().params.checkpoint;
    const saved = (total, messageCount, revision) => ({ promptTokens: total - 10, completionTokens: 10, totalTokens: total, cachedTokens: 0, modelId: model, messageCount, ...(revision ? { revision } : {}) });
    const row = (id, parentId, role, createdAt, metadata = {}) => ({
      id, parentId, role, createdAt, content: [{ type: "text", text: (role === "user" ? "x" : "y").repeat(40) }], metadata,
    });
    const messages = __MESSAGES__;
    let release;
    world.countGate = new Promise((resolve) => { release = resolve; });
    await hydrateThreadUsage({ remoteId: "thread-a", messages });
    const shown = snapshot().contextUsage;
    release();
    await new Promise((resolve) => setTimeout(resolve, 30));
    console.log(JSON.stringify({
      counts: world.countedMessages.length,
      shown: { total: shown?.totalTokens ?? null, estimated: shown?.estimated ?? null },
    }));
"""

_FIRST_TURN = 'row("u1", null, "user", 1), row("a2", "u1", "assistant", 2, __A2__)'


@pytest.mark.parametrize(
    ("a2", "tail", "picked", "expected"),
    [
        (
            "{ contextUsage: saved(300, 2) }",
            ', row("u3", "a2", "user", 3), '
            'row("a4", "u3", "assistant", 4, { contextUsage: saved(500, 4), revision: 1 })',
            "",
            {"counts": 1, "shown": {"total": 320, "estimated": True}},
        ),
        (
            "{ contextUsage: saved(300, 2) }",
            ', row("u3", "a2", "user", 3), '
            'row("a4", "u3", "assistant", 4, { contextUsage: saved(500, 4), revision: 1 })',
            "external::openai::gpt-x",
            {"counts": 0, "shown": {"total": 320, "estimated": True}},
        ),
        (
            "{ contextUsage: saved(300, 2), revision: 1 }",
            ', row("u3", "a2", "user", 3), '
            'row("a4", "u3", "assistant", 10, { contextUsage: saved(500, 4, 1) })',
            "",
            {"counts": 0, "shown": {"total": 500, "estimated": None}},
        ),
        (
            "{ contextUsage: saved(300, 2) }",
            ', row("u5", "a2", "user", 5), row("a6", "u5", "assistant", 6, { contextUsage: saved(700, 6) })',
            "",
            {"counts": 1, "shown": {"total": 320, "estimated": True}},
        ),
        (
            "{ contextUsage: saved(300, 2) }",
            ', row("u3", "a2", "user", 3)',
            "",
            {"counts": 1, "shown": {"total": 310, "estimated": True}},
        ),
    ],
    ids = [
        "local_reply_edited_after_its_count",
        "cloud_reply_edited_after_its_count",
        "reply_written_after_an_earlier_edit",
        "turn_deleted_above_the_reply",
        "unanswered_last_prompt",
    ],
)
def test_history_hydration_trusts_only_a_count_that_still_describes_its_branch(
    a2, tail, picked, expected
):
    pick = f"useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});" if picked else ""
    messages = "[" + _FIRST_TURN.replace("__A2__", a2) + tail + "]"
    out = _run(_script(_HYDRATE_REVISED, PICK = pick, MESSAGES = messages))
    assert out == expected


_OVERLAPPING_RUNS = """
    // @ts-nocheck
    import { renderThreadContextUsageRecount, seed, snapshot, useChatRuntimeStore, world } from "./harness.ts";
    __LOADED_MODEL__
    seed({ loadedIsGguf: true });
    __PICK__
    const model = useChatRuntimeStore.getState().params.checkpoint;
    const msg = (id, role, at, custom = {}) => ({
      id, role, createdAt: new Date(at), content: [{ type: "text", text: (role === "user" ? "x" : "y").repeat(40) }],
      metadata: { custom },
      ...(role === "assistant" ? { status: { type: "complete" } } : {}),
    });
    const saved = (total, messageCount) => ({ promptTokens: total - 10, completionTokens: 10, totalTokens: total, cachedTokens: 0, modelId: model, messageCount });
    const live = [
      msg("u1", "user", 1), msg("a2", "assistant", 2, { contextUsage: saved(300, 2) }),
      msg("u3", "user", 3), msg("a4", "assistant", 4, { contextUsage: saved(500, 4) }),
    ];
    const show = (messages) => {
      world.liveMessages = { "thread-a": messages };
      world.storedMessages["thread-a"] = messages.map((m) => ({ ...m, createdAt: m.createdAt.getTime() }));
    };
    const render = async () => {
      renderThreadContextUsageRecount();
      await new Promise((resolve) => setTimeout(resolve, 30));
    };
    show(live);
    seed({ activeThreadId: "thread-a", runningByThreadId: { "thread-y": true } });
    useChatRuntimeStore.getState().setContextUsage(saved(500, 4));
    await render();
    const countsBefore = world.countedMessages.length;
    __STEPS__
    const usage = snapshot().contextUsage;
    console.log(JSON.stringify({
      counts: world.countedMessages.length - countsBefore,
      total: usage?.totalTokens ?? null,
      estimated: usage?.estimated ?? null,
    }));
"""

_SWITCH_WHILE_ANOTHER_CHAT_RUNS = """
    show([...live.slice(0, 3), msg("a4b", "assistant", 5, { contextUsage: saved(800, 4) })]);
    await render();
    seed({ runningByThreadId: {} });
    await render();
"""

_DELETE_AFTER_OWN_RUN_WHILE_ANOTHER_CHAT_RUNS = """
    seed({ runningByThreadId: { "thread-y": true, "thread-a": true } });
    await render();
    const finished = [...live, msg("u5", "user", 6), msg("a6", "assistant", 7, { contextUsage: saved(700, 6) })];
    show(finished);
    useChatRuntimeStore.getState().setContextUsage(saved(700, 6));
    seed({ runningByThreadId: { "thread-y": true } });
    await render();
    show([finished[0], finished[1], finished[4], finished[5]]);
    await render();
    seed({ runningByThreadId: {} });
    await render();
"""


@pytest.mark.parametrize(
    ("steps", "picked", "expected"),
    [
        (_SWITCH_WHILE_ANOTHER_CHAT_RUNS, "", {"counts": 0, "total": 800, "estimated": None}),
        (
            _SWITCH_WHILE_ANOTHER_CHAT_RUNS,
            "external::openai::gpt-x",
            {"counts": 0, "total": 800, "estimated": None},
        ),
        (
            _DELETE_AFTER_OWN_RUN_WHILE_ANOTHER_CHAT_RUNS,
            "",
            {"counts": 1, "total": 112, "estimated": None},
        ),
        (
            _DELETE_AFTER_OWN_RUN_WHILE_ANOTHER_CHAT_RUNS,
            "external::openai::gpt-x",
            {"counts": 0, "total": 320, "estimated": True},
        ),
    ],
    ids = [
        "local_branch_switch",
        "cloud_branch_switch",
        "local_delete_after_its_own_run",
        "cloud_delete_after_its_own_run",
    ],
)
def test_a_change_made_while_another_chat_generates_is_priced_when_it_ends(steps, picked, expected):
    pick = f"useChatRuntimeStore.getState().setCheckpoint({json.dumps(picked)});" if picked else ""
    out = _run(_script(_OVERLAPPING_RUNS, PICK = pick, STEPS = textwrap.dedent(steps)))
    assert out == expected
