# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opening a New Chat against a resident GGUF must recount the empty prompt (#7450).

A ``/chat?new=<uuid>`` view reaches none of the other recount triggers: it has no
persisted thread for the history loader, and ``ThreadNewChatSwitch`` writes
``setActiveThreadId(null)``, which blanks ``contextUsage``. On a page RELOAD there is a
second gap -- that effect runs before ``/api/inference/status`` answers, so the store
still holds ``checkpoint: ""`` / ``ggufContextLength: null`` and the recount returns
without counting; the component's dependency array is what retries it.

``studio/frontend`` carries no JS test runner, so the effects, the real
``refreshContextUsage`` and the real store reducers are sliced verbatim out of the
studio sources and replayed through a small React-effect emulator (per-effect
dependency arrays, re-run only when a dependency changed) under ``node``.
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
BOUND_NAMES = {"aui", "isLoading", "nonce", "checkpoint", "ggufContextLength", "modelLoading"}


def _refresh_module_body() -> str:
    """Everything in refresh-context-usage.ts after its import block, verbatim."""
    text = read(REFRESH)
    marker = 'from "./chat-history-storage";'
    return text[text.index(marker) + len(marker) :]


def _new_chat_effects() -> list[tuple[list[str], str]]:
    """Every ``useEffect`` in ThreadNewChatSwitch as (dependency names, verbatim body)."""
    component = slice_between(
        read(PROVIDER),
        "function ThreadNewChatSwitch(",
        "\nfunction ActiveThreadSync(",
    )
    matches = re.findall(r"useEffect\(\(\) => \{\n(.*?)\n  \}, \[([^\]]*)\]\);", component, re.S)
    assert matches, "ThreadNewChatSwitch effects not found"
    effects = [
        ([name.strip() for name in deps.split(",") if name.strip()], body) for body, deps in matches
    ]
    for deps, _body in effects:
        unknown = set(deps) - BOUND_NAMES
        assert not unknown, f"emulator does not bind {sorted(unknown)}"
    return effects


def _store_reducers() -> str:
    """setCheckpoint / setActiveThreadId / setContextUsage / setThreadContextUsage, verbatim."""
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
  switchedToNewThread: 0,
  promptQueueStops: 0,
};

const state: any = {
  activeThreadId: null,
  contextUsage: null,
  contextUsageByThreadId: {},
  params: { checkpoint: "", systemPrompt: "", systemVariables: "", maxTokens: 4096 },
  activeGgufVariant: null,
  ggufContextLength: null,
  modelLoading: false,
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

// 12 tokens for the bare template, 25 more per message actually sent.
async function countChatInputTokens(payload: any): Promise<any> {
  world.countedMessages.push(payload.messages);
  return { input_tokens: 12 + payload.messages.length * 25 };
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

// Replays ThreadNewChatSwitch's sliced effects with React's dependency rule: an
// effect re-runs only when one of its own dependencies changed since last render.
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
  effects.forEach((effect: any, index: number) => {
    const next = effect.deps.map((name: string) => scope[name]);
    const previous = renderedDeps[index];
    if (
      previous != null &&
      previous.length === next.length &&
      previous.every((value: any, i: number) => Object.is(value, next[i]))
    ) {
      return;
    }
    renderedDeps[index] = next;
    effect.run();
  });
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


def _rendered_effects() -> str:
    blocks = []
    for deps, body in _new_chat_effects():
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
    render = HARNESS_RENDER.replace("__EFFECTS__", _rendered_effects())
    resident = HARNESS_RESIDENT.replace("__FAST_PATH__", _resident_fast_path())
    return prelude + _refresh_module_body() + render + resident


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script)


# The status response that hydrates a resident GGUF. Neither field is persisted,
# so on a reload both arrive only once /api/inference/status has answered.
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
        # Arriving on New Chat with the GGUF already resident: the store knows the model
        # from the outset, and the usage the chat the user is leaving left behind is
        # blanked. The second render repeats identical store values (the deferred
        # inventory refresh re-writing the same checkpoint) and must not re-price.
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
        # A page RELOAD of /chat?new=<uuid>: neither the checkpoint nor the window is
        # persisted, so the first render sees checkpoint "" / ggufContextLength null and
        # nothing can be priced until /api/inference/status answers.
        pytest.param("", 0, id = "reload_before_status_hydrates"),
    ],
)
def test_a_new_chat_prices_its_empty_prompt_against_a_resident_gguf(
    before_status, expected_early_counts
):
    """#7450: the empty New Chat request already carries a template, but this view reaches
    no other recount trigger, so the bar stays hidden until the first completion. It has to
    land on a count in either order, and open exactly one thread doing so."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
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
    """The user sent a message before the status response landed, so the thread is
    persisted and a real completion owns the bar. The retry effect re-runs whenever the
    model fields change, so it needs its own guard against overwriting that usage."""
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

# A regenerated last answer: the retry is the newest stored leaf, so the branch rebuilt
# from records is the one the runtime is NOT showing.
RETRY_BRANCH_STORED = """
    world.storedMessages["thread-a"] = [
      { id: "m1", role: "user", createdAt: 1, content: [{ type: "text", text: "hi" }], metadata: {} },
      { id: "m2", role: "assistant", createdAt: 2, parentId: "m1", content: [{ type: "text", text: "yo" }], metadata: {} },
      { id: "m3", role: "user", createdAt: 3, parentId: "m2", content: [{ type: "text", text: "again" }], metadata: {} },
      { id: "m4", role: "assistant", createdAt: 4, parentId: "m3", content: [{ type: "text", text: "sure" }], metadata: {} },
    ];
"""

# What the mounted runtime exports for the branch on screen.
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
    ("world_setup", "expected_sent"),
    [
        # No runtime branch published yet (the history loader's own call runs before the
        # import): the stored records are the only source, and both turns must be priced.
        pytest.param(TWO_STORED_TURNS, 2, id = "stored_branch"),
        # A temporary/incognito chat persists nothing at all, so listStoredChatMessages
        # returns [] by design and the records would price a bare template.
        pytest.param(LIVE_INCOGNITO_BRANCH, 3, id = "incognito_thread_stores_nothing"),
        # The user regenerated, then switched back to the first answer. The newest stored
        # leaf is the retry, so records reconstruct four turns the request would not send.
        pytest.param(RETRY_BRANCH_STORED + LIVE_BRANCH, 2, id = "runtime_shows_an_older_branch"),
    ],
)
def test_a_loaded_model_reprices_the_open_thread(world_setup, expected_sent):
    """The post-load recount on a real chat: a model change clears the per-thread cache,
    so the bar has to be refilled by pricing the conversation rather than reusing the last
    completion's usage. It must price the branch the next request would send -- the mounted
    runtime's when it has one, the stored records otherwise -- and it must reach the
    per-thread cache setActiveThreadId restores from, or the bar blanks on the way back."""
    expected_total = 12 + 25 * expected_sent
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
    assert (out["contextUsage"] or {}).get("totalTokens") == expected_total
    assert out["cached"] is not None, "the recount must reach the per-thread cache"
    assert (out["cached"] or {}).get("totalTokens") == expected_total


def test_adopting_the_resident_gguf_reprices_the_open_thread():
    """Switching back from an external provider to the local model that never left memory
    takes loadModel's already-resident branch, which returns before the post-load recount.
    setCheckpoint blanks the bar on the way through and a mounted thread does not rerun its
    history loader, so this branch has to recount or the bar stays empty until the next
    completion."""
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
