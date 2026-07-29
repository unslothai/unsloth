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

TEMP = WORKDIR / "temp" / "new_chat_context_recount"

SOURCES = (REFRESH, PROVIDER, STORE)

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
    """setActiveThreadId / setContextUsage / setThreadContextUsage, verbatim."""
    text = read(STORE)
    active = slice_between(text, "setActiveThreadId: (activeThreadId) =>", "setActiveProjectId:")
    usage = slice_between(text, "setContextUsage: (contextUsage) =>", "setThreadContextUsage:")
    thread_usage = slice_between(text, "setThreadContextUsage: (threadId, usage) =>", "}));")
    return "  " + active.strip() + "\n  " + usage.strip() + "\n  " + thread_usage.strip()


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
  params: { checkpoint: "", systemPrompt: "", systemVariables: "" },
  ggufContextLength: null,
  modelLoading: false,
  artifactsEnabled: false,
  supportsTools: false,
};

function set(updater: any): void {
  const patch = typeof updater === "function" ? updater(state) : updater;
  Object.assign(state, patch);
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
    return prelude + _refresh_module_body() + render


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


def test_new_chat_recounts_against_a_resident_model():
    """The user's previous chat left usage on the bar; New Chat blanks it. With the
    model already resident nothing recounts, so the bar stays hidden even though the
    empty request already carries a template."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            // Usage left behind by the chat the user is leaving.
            seed({{
              activeThreadId: "thread-a",
              contextUsage: {{ promptTokens: 900, completionTokens: 30, totalTokens: 930, cachedTokens: 0 }},
            }});

            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            const after = snapshot();
            console.log(JSON.stringify({{
              switched: world.switchedToNewThread,
              activeThreadId: after.activeThreadId,
              contextUsage: after.contextUsage,
              counts: world.countedMessages.length,
            }}));
            """
        )
    )
    assert out["switched"] == 1, "the effect must still switch to a fresh local thread"
    assert out["activeThreadId"] is None
    assert out["counts"] == 1, "the empty New Chat view must ask the backend for a count"
    assert (
        out["contextUsage"] is not None
    ), "the bar stays hidden on a New Chat opened against an already-loaded GGUF"
    assert out["contextUsage"].get("totalTokens") == 12
    assert out["contextUsage"].get("completionTokens") == 0


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


def test_reloaded_new_chat_recounts_once_status_hydrates():
    """The reload race: the New Chat effect runs against an empty store, then the
    status response fills in the resident model. Something must price the empty
    prompt at that point, and it must not open a second thread doing so."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";

            // Reload of /chat?new=<uuid>: no checkpoint, no window, no thread yet.
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            const beforeHydration = {{
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }};

            // /api/inference/status answers: a GGUF is already resident.
            {LOADED_MODEL}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            const after = snapshot();
            console.log(JSON.stringify({{
              beforeHydration,
              switched: world.switchedToNewThread,
              promptQueueStops: world.promptQueueStops,
              activeThreadId: after.activeThreadId,
              contextUsage: after.contextUsage,
              counts: world.countedMessages.length,
            }}));
            """
        )
    )
    assert (
        out["beforeHydration"].get("counts") == 0
    ), "nothing can be counted before the status response names the model"
    assert out["beforeHydration"].get("contextUsage") is None
    assert out["activeThreadId"] is None
    assert out["counts"] == 1, (
        "a reloaded New Chat view never prices its empty prompt once the "
        "resident GGUF is known, so the bar stays hidden until a completion"
    )
    assert out["contextUsage"] is not None
    assert out["contextUsage"].get("totalTokens") == 12
    assert out["switched"] == 1, "hydration must not open a second thread"
    assert out["promptQueueStops"] == 1, "hydration must not re-stop the prompt queue"


@pytest.mark.parametrize(
    ("first_render_seed", "second_render_seed", "expected_counts", "expected_total"),
    [
        # A later render with the same store values (e.g. the deferred inventory refresh
        # re-writing an identical checkpoint) must not re-price an already-priced bar.
        pytest.param(LOADED_MODEL, "", 1, 12, id = "identical_store_values"),
        # The user sent a message before the status response landed, so the thread is
        # persisted and a real completion owns the bar: leave it alone.
        pytest.param(
            "",
            LOADED_MODEL
            + """
            seed({
              activeThreadId: "thread-a",
              contextUsage: { promptTokens: 640, completionTokens: 40, totalTokens: 680, cachedTokens: 0 },
            });
            """,
            0,
            680,
            id = "completion_owns_the_bar",
        ),
    ],
)
def test_the_hydration_retry_prices_a_blank_bar_only_once(
    first_render_seed, second_render_seed, expected_counts, expected_total
):
    """The second effect re-runs whenever the model fields change, so it needs its own
    guards: one count per New Chat view, and never over a completion's usage."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            {first_render_seed}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            {second_render_seed}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              switched: world.switchedToNewThread,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == expected_counts
    assert out["switched"] == 1, "a re-render must not open a second thread"
    assert out["contextUsage"].get("totalTokens") == expected_total


def test_a_thread_recount_survives_switching_away_and_back():
    """setActiveThreadId restores usage from contextUsageByThreadId only, so a recount
    that landed on the visible thread has to reach that map too. setContextUsage writes
    through for the active thread, which is the thread every recount publish is gated on."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ refreshContextUsage, seed, snapshot, useChatRuntimeStore, world }} from "./harness.ts";
            {LOADED_MODEL}
            world.storedMessages["thread-a"] = [
              {{ id: "m1", role: "user", createdAt: 1, content: [{{ type: "text", text: "hi" }}], metadata: {{}} }},
              {{ id: "m2", role: "assistant", createdAt: 2, content: [{{ type: "text", text: "yo" }}], metadata: {{}} }},
            ];
            // A model change cleared the per-thread cache; the post-load recount refills the bar.
            seed({{ activeThreadId: "thread-a", contextUsage: null, contextUsageByThreadId: {{}} }});
            await refreshContextUsage({{ threadId: "thread-a", afterModelLoad: true }});
            const recounted = snapshot();

            // Switch to another still-mounted thread and back: no history loader reruns.
            useChatRuntimeStore.getState().setActiveThreadId("thread-b");
            const away = snapshot().contextUsage;
            useChatRuntimeStore.getState().setActiveThreadId("thread-a");

            console.log(JSON.stringify({{
              counts: world.countedMessages.length,
              recounted: recounted.contextUsage,
              cached: recounted.contextUsageByThreadId["thread-a"] ?? null,
              away,
              back: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["counts"] == 1
    assert out["recounted"].get("totalTokens") == 62
    assert out["cached"] is not None, "the recount must reach the per-thread cache"
    assert out["cached"].get("totalTokens") == 62
    assert out["away"] is None, "another thread has no usage of its own here"
    assert out["back"].get("totalTokens") == 62, "the bar must not blank on the way back"
