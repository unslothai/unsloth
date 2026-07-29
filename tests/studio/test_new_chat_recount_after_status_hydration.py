# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reloading ``/chat?new=<uuid>`` must still price the empty prompt (#7453).

``ThreadNewChatSwitch`` recounts when a New Chat view opens, but on a page
RELOAD that effect runs before ``/api/inference/status`` has answered: the local
checkpoint and the GGUF window are not persisted, so the store still holds
``checkpoint: ""`` and ``ggufContextLength: null`` and ``refreshContextUsage``
returns without counting. The status hydration that follows only recounts for a
non-null ``activeThreadId``, and this view's thread is deliberately ``null``, so
nothing ever prices the template and the bar stays hidden until a completion.

``ThreadNewChatSwitch``'s effects are sliced verbatim out of the provider and
replayed through a small React-effect emulator (per-effect dependency arrays,
re-run only when a dependency changes), so an effect that never re-runs on the
store update is visible here exactly as it is in the browser.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


REFRESH = _source_path("studio/frontend/src/features/chat/utils/refresh-context-usage.ts")
PROVIDER = _source_path("studio/frontend/src/features/chat/runtime-provider.tsx")
STORE = _source_path("studio/frontend/src/features/chat/stores/chat-runtime-store.ts")
ADAPTER = _source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")

TEMP = WORKDIR / "temp" / "new_chat_recount_after_status_hydration"

SOURCES = (REFRESH, PROVIDER, STORE, ADAPTER)

# Every name the emulator can supply to a sliced dependency array.
BOUND_NAMES = {"aui", "isLoading", "nonce", "checkpoint", "ggufContextLength", "modelLoading"}


def _require_node() -> None:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    for path in SOURCES:
        if not path.exists():
            pytest.skip("studio chat sources not present")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def _slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def _refresh_module_body() -> str:
    """Everything in refresh-context-usage.ts after its import block, verbatim."""
    text = _read(REFRESH)
    marker = 'from "./chat-history-storage";'
    return text[text.index(marker) + len(marker) :]


def _new_chat_effects() -> list[tuple[list[str], str]]:
    """Every ``useEffect`` in ThreadNewChatSwitch as (dependency names, verbatim body)."""
    component = _slice_between(
        _read(PROVIDER),
        "function ThreadNewChatSwitch(",
        "\nfunction ActiveThreadSync(",
    )
    matches = re.findall(
        r"useEffect\(\(\) => \{\n(.*?)\n  \}, \[([^\]]*)\]\);", component, re.S
    )
    assert matches, "ThreadNewChatSwitch effects not found"
    effects = [
        ([name.strip() for name in deps.split(",") if name.strip()], body)
        for body, deps in matches
    ]
    for deps, _body in effects:
        unknown = set(deps) - BOUND_NAMES
        assert not unknown, f"emulator does not bind {sorted(unknown)}"
    return effects


def _store_reducers() -> str:
    """setActiveThreadId / setContextUsage / setThreadContextUsage, verbatim."""
    text = _read(STORE)
    active = _slice_between(text, "setActiveThreadId: (activeThreadId) =>", "setActiveProjectId:")
    usage = _slice_between(text, "setContextUsage: (contextUsage) =>", "setThreadContextUsage:")
    thread_usage = _slice_between(text, "setThreadContextUsage: (threadId, usage) =>", "}));")
    return "  " + active.strip() + "\n  " + usage.strip() + "\n  " + thread_usage.strip()


def _project_resolution() -> str:
    """resolveProjectInstructions + resolveProjectId from the adapter, verbatim."""
    text = _read(ADAPTER)
    instructions = _slice_between(
        text,
        "async function resolveProjectInstructions(",
        "async function resolveChatInstructions(",
    )
    project_id = _slice_between(
        text, "async function resolveProjectId(", "async function resolveSandboxSessionId("
    )
    return instructions.rstrip() + "\n\n" + project_id.rstrip()


HARNESS_PRELUDE = """
// @ts-nocheck
// Fixtures the sliced source reads through. Everything below the PRELUDE marker
// is copied verbatim out of the studio sources.
export const world: any = {
  storedThreads: {} as Record<string, any>,
  storedProjects: {} as Record<string, any>,
  storedMessages: {} as Record<string, any[]>,
  countedMessages: [] as any[][],
  switchedToNewThread: 0,
  promptQueueStops: 0,
};

const state: any = {
  activeThreadId: null,
  activeProjectId: null,
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

async function getStoredChatThread(id: string): Promise<any> {
  return world.storedThreads[id];
}

async function getStoredChatProject(id: string): Promise<any> {
  return world.storedProjects[id];
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

async function buildOutboundMessagesForTokenCount(
  messages: any,
  threadId: string | undefined,
): Promise<any[]> {
  const instructions = await resolveProjectInstructions(threadId);
  const outbound = messages.map((m: any) => ({ role: m.role, content: "x" }));
  if (instructions) {
    outbound.unshift({ role: "system", content: instructions });
  }
  return outbound;
}

async function buildLocalTokenCountExtras(
  _threadId: string | undefined,
  _outbound: any[],
): Promise<Record<string, unknown>> {
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
    return prelude + _project_resolution() + "\n" + _refresh_module_body() + render


def _run(script: str) -> dict:
    _require_node()
    TEMP.mkdir(parents = True, exist_ok = True)
    workdir = Path(tempfile.mkdtemp(prefix = "run", dir = str(TEMP)))
    (workdir / "harness.ts").write_text(_harness_source(), encoding = "utf-8")
    (workdir / "run.mts").write_text(script, encoding = "utf-8")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(workdir),
        capture_output = True,
        text = True,
        timeout = 60,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


# The status response that hydrates a resident GGUF. Neither field is persisted,
# so on a reload both arrive only once /api/inference/status has answered.
HYDRATE_STATUS = """
    seed({
      params: { checkpoint: "unsloth/gguf-model", systemPrompt: "", systemVariables: "" },
      ggufContextLength: 8192,
    });
"""


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
            {HYDRATE_STATUS}
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
    assert out["beforeHydration"]["counts"] == 0, (
        "nothing can be counted before the status response names the model"
    )
    assert out["beforeHydration"]["contextUsage"] is None
    assert out["activeThreadId"] is None
    assert out["counts"] == 1, (
        "a reloaded New Chat view never prices its empty prompt once the "
        "resident GGUF is known, so the bar stays hidden until a completion"
    )
    assert out["contextUsage"] is not None
    assert out["contextUsage"]["totalTokens"] == 12
    assert out["switched"] == 1, "hydration must not open a second thread"
    assert out["promptQueueStops"] == 1, "hydration must not re-stop the prompt queue"


def test_hydration_retry_does_not_double_count_an_already_priced_bar():
    """When the store already has the model, the New Chat effect prices the prompt
    itself; the hydration retry must leave that single count alone."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";
            {HYDRATE_STATUS}
            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));
            // A later render with the same store values (e.g. the deferred inventory
            // refresh re-writing an identical checkpoint).
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
    assert out["counts"] == 1
    assert out["switched"] == 1
    assert out["contextUsage"]["totalTokens"] == 12


def test_hydration_retry_stays_out_of_the_way_of_a_started_chat():
    """The user sent a message before the status response landed: the thread is
    persisted and the completion owns the bar, so the retry must not overwrite it."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ renderNewChatSwitch, seed, snapshot, world }} from "./harness.ts";

            renderNewChatSwitch({{ isLoading: false, nonce: "n1" }});
            await new Promise((resolve) => setTimeout(resolve, 30));

            // First turn completed against the resident model, on a persisted thread.
            {HYDRATE_STATUS}
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
    assert out["contextUsage"]["totalTokens"] == 680
