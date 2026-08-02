# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opening a New Chat against a resident GGUF must recount the empty prompt (#7450).

A ``/chat?new=<uuid>`` view reaches none of the other recount triggers: it has no persisted
thread for the history loader, and ``ThreadNewChatSwitch`` writes ``setActiveThreadId(null)``,
which blanks ``contextUsage``. A page RELOAD adds a second gap -- that effect runs before
``/api/inference/status`` answers, so the store still holds ``checkpoint: ""`` and the recount
returns without counting; the component's dependency array is what retries it.

The effects, the real ``refreshContextUsage`` and the real store reducers are sliced verbatim out
of the studio sources (see ``_node_harness``) and replayed through a React-effect emulator:
per-effect dependency arrays, re-run only when a dependency changed.
"""

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

TEMP = WORKDIR / "temp" / "new_chat_context_recount"

SOURCES = (REFRESH, PROVIDER, STORE, RUNTIME)

# Every name the emulator can supply to a sliced dependency array.
BOUND_NAMES = {
    "activeThreadId",
    "aui",
    "checkpoint",
    "enabled",
    "ggufContextLength",
    "isLoading",
    "runActive",
    "modelLoading",
    "nonce",
}


def _refresh_module_body() -> str:
    """Everything in refresh-context-usage.ts after its import block, verbatim."""
    text = read(REFRESH)
    marker = 'from "./chat-history-storage";'
    return text[text.index(marker) + len(marker) :]


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
        "\n// Exposes the current thread's cancelRun()",
    )


def _store_reducers() -> str:
    """The four reducers the recount goes through, verbatim."""
    text = read(STORE)
    checkpoint = slice_between(
        text,
        "setCheckpoint: (modelId, ggufVariant) =>",
        "  // Re-apply the incoming thread's own usage",
    )
    active = slice_between(text, "setActiveThreadId: (activeThreadId) =>", "setActiveProjectId:")
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


def _resident_fast_path() -> str:
    """loadModel's already-resident branch, verbatim."""
    return slice_between(
        read(RUNTIME),
        "      // Picking an external provider leaves the local model resident",
        "      // Every chat decodes on the llama-server this load replaces",
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
  ggufContextLength: null,
  modelLoading: false,
  runningByThreadId: {},
  // The subset decoding on the local llama-server: the recount must not share it with a decode.
  localRunByThreadId: {},
  // What the recount reads to tell an output-only audio GGUF from a chat one.
  models: [],
};

function set(updater: any): void {
  const patch = typeof updater === "function" ? updater(state) : updater;
  Object.assign(state, patch);
}

// What the sliced setCheckpoint reducer reads through: nothing is persisted here, and
// no external provider is configured, so its output-cap clamp is a no-op.
const CHAT_DEEP_RESEARCH_ENABLED_KEY = "unsloth_deep_research_enabled";
function saveLastExternalCheckpoint(_id: string | null): void {}
function saveBool(_key: string, _value: boolean): void {}
function parseExternalModelId(id: string): any {
  const [providerId, ...rest] = id.split(":");
  return rest.length > 0 ? { providerId, modelId: rest.join(":") } : null;
}
const useExternalProvidersStore: any = { getState: () => ({ providers: [] }) };
function getExternalMaxOutputTokens(_providerType: any, _modelId: any): number {
  return 8192;
}

const actions: any = {
__STORE_REDUCERS__
};

export const useChatRuntimeStore: any = {
  getState: () => ({ ...state, ...actions }),
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

// The adapter's own prompt build is exercised by the request tests; here it only has
// to turn the reconstructed branch into something countable.
async function buildOutboundMessagesForTokenCount(messages: any): Promise<any[]> {
  return messages.map((m: any) => ({ role: m.role, content: "x" }));
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

const auiFixture: any = {
  threads: () => ({
    switchToNewThread: async () => {
      world.switchedToNewThread += 1;
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

export function renderThreadContextUsageRecount(props: any = {}): void {
  const enabled = props.enabled ?? true;
  // Read through the store the way the component's selectors do.
  const activeThreadId = state.activeThreadId;
  const checkpoint = state.params.checkpoint;
  const ggufContextLength = state.ggufContextLength;
  const modelLoading = state.modelLoading;
  const runActive = Object.values(state.runningByThreadId ?? {}).some(Boolean);
  const scope: any = {
    activeThreadId,
    checkpoint,
    enabled,
    ggufContextLength,
    modelLoading,
    runActive,
  };
  const effects: any[] = [
__RECOUNT_EFFECTS__
  ];
  replayEffects(effects, scope, recountDeps);
}

const renderedDeps: any[] = [];

export function renderNewChatSwitch(props: any): void {
  const aui = auiFixture;
  const isLoading = props.isLoading;
  const nonce = props.nonce;
  // The component reads these through useChatRuntimeStore selectors, so a
  // re-render sees whatever the store holds right now.
  const checkpoint = state.params.checkpoint;
  const ggufContextLength = state.ggufContextLength;
  const modelLoading = state.modelLoading;
  const scope: any = {
    aui,
    isLoading,
    nonce,
    checkpoint,
    ggufContextLength,
    modelLoading,
  };
  const effects: any[] = [
__EFFECTS__
  ];
  replayEffects(effects, scope, renderedDeps);
}
"""


HARNESS_RESIDENT = """

// Picking the model that never left memory takes loadModel's already-resident branch,
// sliced verbatim below. It returns early, so it is replayed inside its own function
// with the surrounding load machinery stubbed.
export async function adoptResidentModel(props: any): Promise<void> {
  const forceReload = false;
  const selection = "pick";
  const modelId: string = props.modelId;
  const ggufVariant = props.ggufVariant ?? null;
  const bailIfLoadInFlight = (): boolean => false;
  const applyPerModelConfigToRuntime = (_config: any): void => {};
  const getInferenceStatus = async (): Promise<any> => props.residentStatus;
  const resolveInferenceCheckpointId = (status: any): string | null =>
    status?.active_model ?? null;
  // The real hydration writes the whole status; the recount only reads the window.
  const applyActiveModelStatusToStore = (status: any, _options: any): void => {
    set({
      ggufContextLength: status.is_gguf ? (status.context_length ?? null) : null,
    });
  };
  const syncModelCapabilities = (_id: string, _status: any): void => {};
__FAST_PATH__
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


def _harness_source() -> str:
    prelude = HARNESS_PRELUDE.replace("__STORE_REDUCERS__", _store_reducers())
    render = HARNESS_RENDER.replace("__EFFECTS__", _rendered_effects(_new_chat_effects())).replace(
        "__RECOUNT_EFFECTS__", _rendered_effects(_thread_recount_effects())
    )
    resident = HARNESS_RESIDENT.replace("__FAST_PATH__", _resident_fast_path())
    return prelude + _refresh_module_body() + render + resident


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script)


# The status response that hydrates a resident GGUF; neither field survives a reload.
LOADED_MODEL = """
    seed({
      params: { checkpoint: "unsloth/gguf-model", systemPrompt: "", systemVariables: "" },
      ggufContextLength: 8192,
      modelLoading: false,
    });
"""


@pytest.mark.parametrize(
    ("before_status", "expected_early_counts"),
    [
        # Arriving on New Chat with the GGUF already resident: the model is known from the outset
        # and the outgoing chat's usage is blanked. The second render repeats identical store
        # values (a deferred inventory refresh rewriting the checkpoint) and must not re-price.
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
        # A page RELOAD of /chat?new=<uuid>: nothing is priceable until /api/inference/status answers.
        pytest.param("", 0, id = "reload_before_status_hydrates"),
        # New Chat opened FROM a populated conversation, which is left running: its runtime stays
        # mounted and the live branch reader keeps returning its messages until the voided
        # switchToNewThread() settles. The empty chat must still be priced as a bare template.
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
      ggufContextLength: null,
    });
"""


@pytest.mark.parametrize(
    ("seed_script", "is_loading", "expected_switched"),
    [
        # No GGUF window means nothing to price; the seeded placeholder must not stick.
        pytest.param(NO_LOCAL_MODEL, "false", 1, id = "no_local_model"),
        # assistant-ui is still hydrating: both effects bail before touching anything.
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

# A regenerated last answer: the newest stored leaf is not the branch the runtime is showing.
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
        # No runtime branch yet: the stored records are the only source, and both turns count.
        pytest.param(TWO_STORED_TURNS, 2, None, id = "stored_branch"),
        # An incognito chat persists nothing, so the records would price a bare template.
        pytest.param(LIVE_INCOGNITO_BRANCH, 3, None, id = "incognito_thread_stores_nothing"),
        # Regenerated, then switched back: the stored leaf is the retry, four turns not sent.
        pytest.param(
            RETRY_BRANCH_STORED + LIVE_BRANCH, 2, None, id = "runtime_shows_an_older_branch"
        ),
        # The endpoint counts with whatever is resident, never the model asked for. Another tab
        # loaded a different GGUF, so the total came from a tokenizer whose window the bar is not
        # showing -- and this client's checkpoint never moved, so the reported id is the witness.
        pytest.param(
            TWO_STORED_TURNS, 2, "unsloth/other-gguf", id = "another_client_swapped_the_model"
        ),
    ],
)
def test_a_loaded_model_reprices_the_open_thread(world_setup, expected_sent, counted_model):
    """The post-load recount on a real chat: a model change clears the per-thread cache, so the bar
    has to be refilled by pricing the conversation. It must price the branch the next request would
    send -- the mounted runtime's when it has one, the stored records otherwise -- and reach the
    per-thread cache setActiveThreadId restores from, or the bar blanks on the way back. A total
    counted by another tokenizer is dropped instead, leaving the previous usage in place."""
    # None means the reply names the model this client already holds, so it is published.
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
        # Sent mid-count then stopped before any usage, so the snapshot guard cannot see the turn.
        pytest.param(True, None, id = "a_turn_arrives_mid_count"),
        # Control: the branch the count priced is still the one on screen.
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
        # A run that BEGINS after the count was issued. The entry gate cannot catch this one: it
        # ran when the thread was idle, so only the publish guard is left to drop the total.
        (True, True, None),
        # Stopped before the count returned, so runningByThreadId is already false and the
        # usage snapshot is still equal: only the content makes the branch look different.
        (False, True, None),
        (False, False, 62),
    ],
    ids = ["run_starts_mid_count", "stopped_before_publish", "idle_and_unchanged"],
)
def test_a_count_taken_while_the_thread_is_running_is_dropped(running, grew, expected_total):
    """A run streaming into an existing turn grows its content without moving the branch length or
    its last id, so the partial is what got priced. The run writes its own usage when it lands.

    Seeded idle and flipped mid-count on purpose: seeding it running would now be refused before
    the request went out, which would exercise the entry gate instead of this guard."""
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
        # A tool result landing on an existing tool-call part: no `text`, and no part added.
        ("live[1].content[0] = { ...live[1].content[0], result: { rows: 4000 } };", None),
        # An edit to different text of the same length, which a size-based signature cannot see.
        ('live[0].content = [{ type: "text", text: "ih" }];', None),
        # Deleting an attachment. The counter prices attachment text, but the removal handler
        # rewrites `attachments` alone and leaves content and ids as they were.
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
              ggufContextLength: null,
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


# Revisiting a thread that was NOT open when the model changed: setCheckpoint emptied
# contextUsageByThreadId, and the thread is already mounted so its history loader never reruns.
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

# A deep link to /chat/:id against a resident GGUF: the history loader's own recount runs before
# /api/inference/status answers, and status lands before ThreadAutoSwitch writes activeThreadId.
DEEP_LINK_HYDRATING_AFTER_THE_LOADER = """
    renderThreadContextUsageRecount();
    await refreshContextUsage({ threadId: "thread-a" });
    const beforeSwitch = world.countedMessages.length;

    // /api/inference/status answers while the thread is still not active.
    seed({
      params: { checkpoint: "unsloth/gguf-model", systemPrompt: "", systemVariables: "" },
      ggufContextLength: 8192,
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
    # thread-b still holds a completion's usage: exact, so becoming active must leave it alone.
    assert out["beforeSwitch"] == 0, "nothing to reprice until the thread is the active one"
    assert (out["contextUsage"] or {}).get("totalTokens") == 62, (
        "a thread the bar points at with no cached usage stays blank until the next "
        "completion unless becoming active reprices it"
    )
    assert (out["cached"] or {}).get("totalTokens") == 62
    assert out["counts"] == 1
    assert len(out["sent"]) == 2, "the thread's stored branch must be priced"


@pytest.mark.parametrize(
    ("mount", "expected_total"),
    [
        # The runtime mounted mid-count after a turn was sent, so the priced branch is a prefix.
        (
            'live = [...stored, { id: "m3", role: "user", createdAt: new Date(3),'
            ' content: [{ type: "text", text: "and another" }] }];',
            None,
        ),
        # Mounted with the same branch: nothing moved, so the stored count still describes it.
        ("live = stored.slice();", 62),
        # Never mounted: there is nothing to compare against and nothing to invalidate.
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
        # Output only: every send goes to /audio/generate instead of a chat completion.
        ("{ isAudio: true, hasAudioInput: false }", 0),
        # Audio IN, chat out: a normal completion, so the chat-template total is the right one.
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
        # Decoding on the local llama-server: the count would share the process with generation.
        ('{ "thread-a": true }', 0),
        # A different thread, still the same llama-server.
        ('{ "thread-b": true }', 0),
        # Control: an idle server is what the count is for.
        ("{}", 1),
    ],
    ids = ["this_thread_running", "another_thread_running", "nothing_running"],
)
def test_no_count_is_issued_while_anything_is_generating(local_runs, expected_counts):
    """/apply-template and /tokenize take no inference slot, so the measured cost of counting
    during a decode is inside the noise, but the budget for this endpoint is zero rather than
    small. The request is not issued at all while a run is live.

    Every run, not only the local ones. An external-provider run cannot contend for llama-server,
    but chat_count_tokens refuses during one regardless, because state's active_generations does
    not distinguish them. Gating on less than the server refuses on would spend a request to be
    told 503 and then never retry, since only what this effect depends on can re-fire it."""
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
        "\n// Exposes the current thread's cancelRun()",
    )
    assert "runningByThreadId" in recount, "the effect must observe decoding"
    deps = re.search(r"\}, \[([^\]]*)\]\);", recount, re.S)
    assert deps and "runActive" in deps.group(1), (
        "runActive must be a DEPENDENCY, not just a guard: nothing else in this array "
        "changes when a run ends, so without it a skipped count would never be retried"
    )
