# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opening a New Chat against a resident GGUF must recount the empty prompt (#7453).

The bar this PR adds is fed by three triggers: the history loader (needs a
persisted thread), the startup hydration (needs a non-null ``activeThreadId``)
and the post-load recount. A ``/chat?new=<uuid>`` view has none of them:
``ThreadNewChatSwitch`` switches to a local thread and writes
``setActiveThreadId(null)``, which blanks ``contextUsage``, and
``ActiveThreadSync`` is disabled while ``newThreadNonce`` is present. So loading
a model on that view shows a bar, while arriving on it with the same model
already loaded shows nothing until the first completion.

The real ``refreshContextUsage``, the real ``ThreadNewChatSwitch`` effect body,
the real store reducers and the real project resolution are sliced verbatim into
a node harness, since ``studio/frontend`` carries no JS test runner.
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

TEMP = WORKDIR / "temp" / "new_chat_context_recount"

SOURCES = (REFRESH, PROVIDER, STORE, ADAPTER)


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
    return text[text.index(marker) + len(marker):]


def _new_chat_effect_body() -> str:
    """The body of ThreadNewChatSwitch's effect, verbatim."""
    text = _read(PROVIDER)
    match = re.search(
        r"function ThreadNewChatSwitch\(.*?useEffect\(\(\) => \{\n(.*?)\n  \}, \[aui, isLoading, nonce\]\);",
        text,
        re.S,
    )
    assert match, "ThreadNewChatSwitch effect not found"
    return match.group(1)


def _store_reducers() -> str:
    """setActiveThreadId / setContextUsage / setThreadContextUsage, verbatim."""
    text = _read(STORE)
    active = _slice_between(
        text, "setActiveThreadId: (activeThreadId) =>", "setActiveProjectId:"
    )
    usage = _slice_between(
        text, "setContextUsage: (contextUsage) =>", "setThreadContextUsage:"
    )
    thread_usage = _slice_between(
        text, "setThreadContextUsage: (threadId, usage) =>", "}));"
    )
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

// Stands in for the real builder: it forwards the same threadId to the verbatim
// project resolution below, which is the part under test here.
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

// ---- PRELUDE ENDS: verbatim studio source follows ----
"""

HARNESS_EFFECT = """

// Verbatim body of ThreadNewChatSwitch's effect.
const requestPromptQueueStop = (_opts: any): void => {
  world.promptQueueStops += 1;
};

const aui: any = {
  threads: () => ({
    switchToNewThread: async () => {
      world.switchedToNewThread += 1;
    },
  }),
};

export function runNewChatSwitchEffect(isLoading: boolean): void {
__EFFECT_BODY__
}
"""


def _harness_source() -> str:
    prelude = HARNESS_PRELUDE.replace("__STORE_REDUCERS__", _store_reducers())
    effect = HARNESS_EFFECT.replace("__EFFECT_BODY__", _new_chat_effect_body())
    return prelude + _project_resolution() + "\n" + _refresh_module_body() + effect


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
            import {{ runNewChatSwitchEffect, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            // Usage left behind by the chat the user is leaving.
            seed({{
              activeThreadId: "thread-a",
              contextUsage: {{ promptTokens: 900, completionTokens: 30, totalTokens: 930, cachedTokens: 0 }},
            }});

            runNewChatSwitchEffect(false);
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
    assert out["contextUsage"] is not None, (
        "the bar stays hidden on a New Chat opened against an already-loaded GGUF"
    )
    assert out["contextUsage"]["totalTokens"] == 12
    assert out["contextUsage"]["completionTokens"] == 0


def test_new_chat_inside_a_project_prices_the_project_instructions():
    """A fresh chat in a project has no thread record, so the recount resolves its
    project through the active-project branch rather than a thread lookup."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ runNewChatSwitchEffect, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            world.storedProjects["p1"] = {{ id: "p1", archived: false, instructions: "Answer in French." }};
            seed({{ activeProjectId: "p1" }});

            runNewChatSwitchEffect(false);
            await new Promise((resolve) => setTimeout(resolve, 30));

            console.log(JSON.stringify({{
              contextUsage: snapshot().contextUsage,
              sent: world.countedMessages.at(-1) ?? [],
            }}));
            """
        )
    )
    assert out["contextUsage"] is not None
    assert any(
        message["role"] == "system" for message in out["sent"]
    ), "the project's instructions must be part of the counted prompt"
    assert out["contextUsage"]["totalTokens"] == 37


def test_new_chat_leaves_the_bar_hidden_with_no_local_model():
    """No GGUF window means nothing to price; the seeded placeholder must not stick."""
    out = _run(
        textwrap.dedent(
            """
            // @ts-nocheck
            import { runNewChatSwitchEffect, seed, snapshot, world } from "./harness.ts";
            seed({
              params: { checkpoint: "", systemPrompt: "", systemVariables: "" },
              ggufContextLength: null,
            });

            runNewChatSwitchEffect(false);
            await new Promise((resolve) => setTimeout(resolve, 30));

            console.log(JSON.stringify({
              contextUsage: snapshot().contextUsage,
              counts: world.countedMessages.length,
            }));
            """
        )
    )
    assert out["counts"] == 0
    assert out["contextUsage"] is None


def test_a_loading_switch_does_not_recount():
    """assistant-ui is still hydrating: the effect bails before touching anything."""
    out = _run(
        textwrap.dedent(
            f"""
            // @ts-nocheck
            import {{ runNewChatSwitchEffect, seed, snapshot, world }} from "./harness.ts";
            {LOADED_MODEL}
            runNewChatSwitchEffect(true);
            await new Promise((resolve) => setTimeout(resolve, 30));

            console.log(JSON.stringify({{
              switched: world.switchedToNewThread,
              counts: world.countedMessages.length,
              contextUsage: snapshot().contextUsage,
            }}));
            """
        )
    )
    assert out["switched"] == 0
    assert out["counts"] == 0
    assert out["contextUsage"] is None
