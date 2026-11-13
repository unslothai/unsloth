// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import type { ThreadMessage } from "@assistant-ui/react";

import * as messageOrder from "../src/features/chat/utils/message-order.ts";
import * as contextUsageReasoning from "../src/features/chat/utils/context-usage-reasoning.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Usage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cachedTokens: number;
  estimated?: boolean;
};

type RefreshModule = {
  refreshContextUsage: (options?: { invalidate?: boolean; afterModelLoad?: boolean }) => Promise<void>;
  setActiveBranchReader: (
    reader: () => readonly ThreadMessage[] | null,
  ) => void;
};

function message(
  id: string,
  role: "user" | "assistant",
  text: string,
): ThreadMessage {
  const base = { id, createdAt: new Date(0), content: [{ type: "text" as const, text }] };
  return role === "user"
    ? { ...base, role, attachments: [], metadata: { custom: {} } }
    : {
        ...base,
        role,
        status: { type: "complete", reason: "stop" },
        metadata: {
          custom: {},
          steps: [],
          unstable_annotations: [],
          unstable_data: [],
          unstable_state: null,
        },
      };
}

function deferred() {
  let release = () => {};
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

function harness() {
  const state = {
    activeThreadId: "thread-1",
    activeLoadId: "load-1",
    contextUsage: null as Usage | null,
    loadedContextLength: 4096,
    loadedIsGguf: true,
    modelLoading: false,
    mcpEnabledForChat: false,
    models: [],
    params: { checkpoint: "local-model", temperature: 0.7 },
    queuedSettingsEpoch: 0,
    runningByThreadId: {} as Record<string, boolean>,
    setContextUsage: (usage: Usage | null) => {
      state.contextUsage = usage;
    },
  };
  const world = {
    branch: [
      message("u1", "user", "question"),
      message("a1", "assistant", "answer"),
    ],
    stored: [message("deleted", "assistant", "deleted reply")],
    extras: {} as Record<string, unknown>,
    historyFields: {} as Record<string, unknown>,
    reasoning: { enable_thinking: false },
    requests: [] as Record<string, unknown>[],
    gate: null as Promise<void> | null,
    historyGate: null as Promise<void> | null,
    storedGate: null as Promise<void> | null,
    mcpEpoch: 0,
    failure: false,
    tokens: 42,
  };
  const refresh = loadWithStubs<RefreshModule>(
    new URL(
      "../src/features/chat/utils/refresh-context-usage.ts",
      import.meta.url,
    ),
    {
      "../api/chat-adapter": {
        buildLocalTokenCountHistory: async (
          messages: readonly ThreadMessage[],
        ) => {
          await world.historyGate;
          return {
            messages: messages.map(({ role, content }) => ({ role, content })),
            ...world.historyFields,
          };
        },
        buildLocalTokenCountExtras: async () => world.extras,
        buildLocalTokenCountReasoning: () => world.reasoning,
        findLatestUserAudioBase64: () => null,
        findLatestUserVideoBase64: () => null,
        messagesContainImage: () => false,
      },
      "../api/chat-api": {
        countChatInputTokens: async (payload: Record<string, unknown>, serializedPayload: string) => {
          assert.deepEqual(JSON.parse(serializedPayload), payload);
          world.requests.push(payload);
          const input_tokens = world.tokens;
          const failure = world.failure;
          await world.gate;
          if (failure) throw new Error("count unavailable");
          return { input_tokens, model: "local-model" };
        },
      },
      "../external-providers": { isExternalModelId: (id: string) => id.startsWith("external:") },
      "../api/mcp-server-mutation-tracker": { getMcpServerMutationEpoch: () => world.mcpEpoch },
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => state },
      },
      "./branch-head": { savedBranchHead: () => undefined },
      "./context-usage-reasoning": contextUsageReasoning,
      "./chat-history-storage": {
        listStoredChatMessages: async () => {
          await world.storedGate;
          return world.stored;
        },
      },
      "./message-order": messageOrder,
    },
  );
  refresh.setActiveBranchReader(() => world.branch);
  return { state, world, ...refresh };
}

test("the context recount forwards the tool-history request fields", async () => {
  const h = harness();
  h.world.historyFields = { studio_tool_history: true };
  await h.refreshContextUsage();
  assert.equal(h.world.requests[0]?.studio_tool_history, true);
  assert.deepEqual(
    h.world.requests[0]?.messages,
    h.world.branch.map(({ role, content }) => ({ role, content })),
  );
});

test("changed message identity and compaction boundaries invalidate identical wire history", async () => {
  const h = harness();
  await h.refreshContextUsage();
  h.world.branch[1] = message("a2", "assistant", "answer");
  await h.refreshContextUsage();
  h.world.branch[1] = {
    ...h.world.branch[1],
    metadata: { custom: { contextTruncation: { boundary_messages: 2 } } },
  } as ThreadMessage;
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 3);
  assert.deepEqual(h.world.requests[0], h.world.requests[2]);
});

test("successive triggers share a completed count after an estimate is restored", async () => {
  const h = harness();
  await h.refreshContextUsage();
  await h.refreshContextUsage();
  h.state.contextUsage = {
    ...h.state.contextUsage!,
    totalTokens: 20,
    estimated: true,
  };
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 1);
  assert.equal(h.state.contextUsage?.totalTokens, 42);
  assert.equal(h.state.contextUsage?.estimated, undefined);
});

test("request changes and explicit invalidation each obtain a new count", async () => {
  const h = harness();
  await h.refreshContextUsage();
  h.world.reasoning = { enable_thinking: true };
  await h.refreshContextUsage();
  h.world.extras = { enable_tools: true, context_overflow: "compact" };
  await h.refreshContextUsage();
  await h.refreshContextUsage({ invalidate: true });
  assert.equal(h.world.requests.length, 4);
});

test("an old reasoning count is discarded and the queued setting is counted", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  const first = h.refreshContextUsage();
  await setImmediate();
  h.world.reasoning = { enable_thinking: true };
  h.world.tokens = 52;
  await h.refreshContextUsage({ invalidate: true });
  gate.release();
  await first;
  await setImmediate();
  assert.deepEqual(
    h.world.requests.map((request) => request.enable_thinking),
    [false, true],
  );
  assert.equal((h.state.contextUsage as Usage | null)?.totalTokens, 52);
});

for (const invalidate of [false, true]) {
  test(`a sampling change during a count reuses the result (invalidate=${invalidate})`, async () => {
    const h = harness();
    const gate = deferred();
    h.world.gate = gate.promise;
    const first = h.refreshContextUsage({ invalidate });
    await setImmediate();
    h.state.params.temperature = 0.9;
    h.state.queuedSettingsEpoch += 1;
    gate.release();
    await first;
    await setImmediate();
    assert.equal(h.world.requests.length, 1);
    assert.equal(h.state.contextUsage?.totalTokens, 42);
  });
}

test("a prompt setting change during a count obtains a replacement without another trigger", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  const first = h.refreshContextUsage();
  await setImmediate();
  h.world.extras = { enable_tools: true };
  h.world.tokens = 52;
  h.state.queuedSettingsEpoch += 1;
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 2);
  assert.equal(h.world.requests[1]?.enable_tools, true);
  assert.equal((h.state.contextUsage as Usage | null)?.totalTokens, 52);
});

test("a settings change while building history restarts the count before sending", async () => {
  const h = harness();
  const gate = deferred();
  h.world.historyGate = gate.promise;
  const first = h.refreshContextUsage();
  h.state.queuedSettingsEpoch += 1;
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 1);
  assert.equal(h.state.contextUsage?.totalTokens, 42);
});

test("a failed count does not replay an identical overlapping trigger", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  h.world.failure = true;
  const first = h.refreshContextUsage();
  await setImmediate();
  await h.refreshContextUsage();
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 1);
  assert.equal(h.state.contextUsage, null);
});

test("a failed count still retries queued reasoning changes", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  h.world.failure = true;
  const first = h.refreshContextUsage();
  await setImmediate();
  h.world.reasoning = { enable_thinking: true };
  h.world.failure = false;
  await h.refreshContextUsage({ invalidate: true });
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 2);
  assert.equal(h.state.contextUsage?.totalTokens, 42);
});

test("a failed count still retries a changed branch without explicit invalidation", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  h.world.failure = true;
  const first = h.refreshContextUsage();
  await setImmediate();
  h.world.branch = [message("u2", "user", "new question")];
  h.world.failure = false;
  await h.refreshContextUsage();
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 2);
  assert.equal(h.state.contextUsage?.totalTokens, 42);
});

test("a completed request cache preserves a newer exact generation count", async () => {
  const h = harness();
  await h.refreshContextUsage();
  const completed = { ...h.state.contextUsage!, totalTokens: 80 };
  h.state.contextUsage = completed;
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 1);
  assert.equal(h.state.contextUsage, completed);
});

test("a cached count cannot replace usage published while history was resolving", async () => {
  const h = harness();
  await h.refreshContextUsage();
  const gate = deferred();
  h.world.historyGate = gate.promise;
  const second = h.refreshContextUsage();
  const completed = { ...h.state.contextUsage!, totalTokens: 80 };
  h.state.contextUsage = completed;
  gate.release();
  await second;
  assert.equal(h.world.requests.length, 1);
  assert.equal(h.state.contextUsage, completed);
});

test("an emptied mounted branch never falls back to deleted stored messages", async () => {
  const h = harness();
  h.world.branch = [];
  await h.refreshContextUsage();
  assert.deepEqual(h.world.requests[0]?.messages, []);
});

for (const loadedIsGguf of [true, false]) {
  test(`tool-enabled counts use the server's result (GGUF=${loadedIsGguf})`, async () => {
    const h = harness();
    h.state.loadedIsGguf = loadedIsGguf;
    h.world.branch = [message("tip", "user", "text")];
    h.world.extras = {
      enable_tools: true,
      permission_mode: "auto",
      rag_scope: { project_id: "project-1", autoinject: true },
    };
    await h.refreshContextUsage();
    assert.equal(h.world.requests.length, 1);
    assert.equal(h.state.contextUsage?.totalTokens, 42);
  });

  test(`a refused tool-enabled count preserves the estimate (GGUF=${loadedIsGguf})`, async () => {
    const h = harness();
    h.state.loadedIsGguf = loadedIsGguf;
    h.world.branch = [message("tip", "user", "text")];
    h.world.extras = {
      enable_tools: true,
      permission_mode: "auto",
      rag_scope: { project_id: "project-1", autoinject: true },
    };
    h.world.failure = true;
    const estimate = {
      promptTokens: 10,
      completionTokens: 0,
      totalTokens: 10,
      cachedTokens: 0,
      estimated: true,
    };
    h.state.contextUsage = estimate;
    await h.refreshContextUsage();
    assert.equal(h.world.requests.length, 1);
    assert.equal(h.state.contextUsage, estimate);
  });
}

test("negative count responses never become exact usage", async () => {
  const h = harness();
  h.world.tokens = -1;
  await h.refreshContextUsage();
  assert.equal(h.state.contextUsage, null);
});

for (const guard of ["external", "loading"]) {
  test(`explicit invalidation clears the cache during ${guard}`, async () => {
    const h = harness();
    await h.refreshContextUsage();
    h.state.params.checkpoint = guard === "external" ? "external:model" : "local-model";
    h.state.modelLoading = guard === "loading";
    await h.refreshContextUsage({ invalidate: true });
    h.state.params.checkpoint = "local-model";
    h.state.modelLoading = false;
    h.state.contextUsage = null;
    h.world.tokens = 52;
    await h.refreshContextUsage();
    assert.equal(h.world.requests.length, 2);
    assert.equal((h.state.contextUsage as Usage | null)?.totalTokens, 52);
  });
}

test("a model reload counts again even with identical configuration", async () => {
  const h = harness();
  await h.refreshContextUsage();
  h.world.tokens = 52;
  await h.refreshContextUsage({ afterModelLoad: true });
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 2);
  assert.equal((h.state.contextUsage as Usage | null)?.totalTokens, 52);
});

test("overlapping post-load triggers share the same count", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  const first = h.refreshContextUsage({ afterModelLoad: true });
  await setImmediate();
  await h.refreshContextUsage({ afterModelLoad: true });
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 1);
  assert.equal(h.state.contextUsage?.totalTokens, 42);
});

test("a new load supersedes an in-flight post-load count", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  const first = h.refreshContextUsage({ afterModelLoad: true });
  await setImmediate();
  h.state.activeLoadId = "load-2";
  h.world.tokens = 52;
  await h.refreshContextUsage({ afterModelLoad: true });
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 2);
  assert.equal(h.state.contextUsage?.totalTokens, 52);
});

test("MCP server mutations invalidate counts with otherwise identical requests", async () => {
  const h = harness();
  h.state.mcpEnabledForChat = true;
  h.world.extras = { enable_tools: true, mcp_enabled: true };
  h.world.branch = [];
  await h.refreshContextUsage();
  h.world.mcpEpoch += 1;
  h.state.contextUsage = null;
  h.world.tokens = 52;
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 2);
  assert.deepEqual(h.world.requests[0], h.world.requests[1]);
  assert.equal((h.state.contextUsage as Usage | null)?.totalTokens, 52);
});

test("MCP changes during an in-flight count cannot publish the old schemas", async () => {
  const h = harness();
  const gate = deferred();
  h.state.mcpEnabledForChat = true;
  h.world.extras = { enable_tools: true, mcp_enabled: true };
  h.world.gate = gate.promise;
  const first = h.refreshContextUsage();
  await setImmediate();
  h.world.mcpEpoch += 1;
  h.world.tokens = 52;
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 2);
  assert.equal((h.state.contextUsage as Usage | null)?.totalTokens, 52);
});

test("a plain queued trigger cannot undo an in-flight explicit invalidation", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  const first = h.refreshContextUsage();
  await setImmediate();
  h.world.tokens = 52;
  await h.refreshContextUsage({ invalidate: true });
  await h.refreshContextUsage();
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 2);
  assert.equal((h.state.contextUsage as Usage | null)?.totalTokens, 52);
});

test("exact usage restored during storage loading prevents a count and settings replay", async () => {
  const h = harness();
  const gate = deferred();
  h.setActiveBranchReader(() => null);
  h.world.storedGate = gate.promise;
  const first = h.refreshContextUsage();
  await setImmediate();
  h.state.queuedSettingsEpoch += 1;
  const restored = { promptTokens: 40, completionTokens: 2, totalTokens: 42, cachedTokens: 0 };
  h.state.contextUsage = restored;
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 0);
  assert.equal(h.state.contextUsage, restored);
});

test("an overlapping trigger cannot replace a restored exact count", async () => {
  const h = harness();
  const gate = deferred();
  h.world.gate = gate.promise;
  const first = h.refreshContextUsage();
  await setImmediate();
  await h.refreshContextUsage();
  h.state.queuedSettingsEpoch += 1;
  const restored = { promptTokens: 40, completionTokens: 2, totalTokens: 42, cachedTokens: 0 };
  h.state.contextUsage = restored;
  gate.release();
  await first;
  await setImmediate();
  assert.equal(h.world.requests.length, 1);
  assert.equal(h.state.contextUsage, restored);
});

test("compacted chats are not recounted by backends without boundary replay", async () => {
  const h = harness();
  h.state.loadedIsGguf = false;
  h.world.branch[1] = {
    ...h.world.branch[1],
    metadata: { custom: { contextTruncation: { boundary_messages: 2 } } },
  } as ThreadMessage;
  const restored = { promptTokens: 40, completionTokens: 2, totalTokens: 42, cachedTokens: 0 };
  h.state.contextUsage = restored;
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 0);
  assert.equal(h.state.contextUsage, restored);
});

test("a changed injected system prompt invalidates identical branch content", async () => {
  const h = harness();
  await h.refreshContextUsage();
  h.world.historyFields = {
    messages: [{ role: "system", content: "updated project instructions" }],
  };
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 2);
});
