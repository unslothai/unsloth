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
  refreshContextUsage: (options?: { invalidate?: boolean }) => Promise<void>;
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
    contextUsage: null as Usage | null,
    loadedContextLength: 4096,
    loadedIsGguf: true,
    modelLoading: false,
    models: [],
    params: { checkpoint: "local-model" },
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
        countChatInputTokens: async (payload: Record<string, unknown>) => {
          world.requests.push(payload);
          const input_tokens = world.tokens;
          const failure = world.failure;
          await world.gate;
          if (failure) throw new Error("count unavailable");
          return { input_tokens, model: "local-model" };
        },
      },
      "../external-providers": { isExternalModelId: () => false },
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => state },
      },
      "./branch-head": { savedBranchHead: () => undefined },
      "./context-usage-reasoning": contextUsageReasoning,
      "./chat-history-storage": {
        listStoredChatMessages: async () => world.stored,
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
  assert.equal(h.state.contextUsage?.totalTokens, 52);
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

for (const role of ["user", "assistant"] as const) {
  test(`GGUF RAG recount ${role === "user" ? "declines a pending" : "prices a completed"} turn`, async () => {
    const h = harness();
    h.world.branch = [message("tip", role, "text")];
    h.world.extras = {
      rag_scope: { project_id: "project-1", autoinject: false },
    };
    await h.refreshContextUsage();
    assert.equal(h.world.requests.length, role === "user" ? 0 : 1);
  });
}

for (const permission of ["auto", "ask"] as const) {
  test(`MLX RAG recount respects ${permission} retrieval permission`, async () => {
    const h = harness();
    h.state.loadedIsGguf = false;
    h.world.branch = [message("tip", "user", "text")];
    h.world.extras = {
      enable_tools: true,
      permission_mode: permission,
      rag_scope: { project_id: "project-1", autoinject: true },
    };
    await h.refreshContextUsage();
    assert.equal(h.world.requests.length, permission === "auto" ? 0 : 1);
  });
}

test("MLX RAG recount prices a turn with retrieval disabled", async () => {
  const h = harness();
  h.state.loadedIsGguf = false;
  h.world.branch = [message("tip", "user", "text")];
  h.world.extras = {
    enable_tools: true,
    rag_scope: { thread_id: "thread-1", autoinject: false, whole_doc: false },
  };
  await h.refreshContextUsage();
  assert.equal(h.world.requests.length, 1);
});

test("negative count responses never become exact usage", async () => {
  const h = harness();
  h.world.tokens = -1;
  await h.refreshContextUsage();
  assert.equal(h.state.contextUsage, null);
});
