// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /context: the command match, the breakdown view, and that the command only counts, so the
// conversation and the next completion (and with them the prompt cache) are untouched.

import assert from "node:assert/strict";
import test from "node:test";
import {
  deriveContextView,
  isContextCommand,
  usagePercent,
} from "../src/features/chat/utils/context-command.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

test("only the bare /context command matches", () => {
  for (const text of ["/context", "  /context\n", "/CONTEXT", "/Context "]) {
    assert.equal(isContextCommand(text), true, JSON.stringify(text));
  }
  for (const text of [
    "",
    "/context please",
    "context",
    "/contexts",
    "what is /context",
    "/ context",
  ]) {
    assert.equal(isContextCommand(text), false, JSON.stringify(text));
  }
});

const breakdown = {
  system: 400,
  tools: 1600,
  messages: 2000,
  tool_names: ["web_search", "python"],
};

test("the view splits the window into the counted parts and what is left", () => {
  const view = deriveContextView(
    {
      input_tokens: 4000,
      breakdown,
      context_length: 16000,
      compaction_budget: 12000,
    },
    null,
  );
  assert.ok(view);
  assert.equal(view.used, 4000);
  assert.equal(view.window, 16000);
  assert.equal(view.percent, 25);
  assert.equal(view.free, 12000);
  assert.deepEqual(
    view.segments.map((s) => [s.key, s.tokens]),
    [
      ["system", 400],
      ["tools", 1600],
      ["messages", 2000],
    ],
  );
  assert.deepEqual(view.compactAt, { tokens: 12000, share: 0.75 });
  assert.deepEqual(view.toolNames, ["web_search", "python"]);
  assert.equal(view.prompt, null);
});

test("the view falls back to the loaded window and shows no marker when nothing compacts", () => {
  const view = deriveContextView(
    { input_tokens: 4000, breakdown, compaction_budget: null },
    8000,
  );
  assert.equal(view?.window, 8000);
  assert.equal(view?.compactAt, null);
});

test("an overfull prompt caps at the window instead of overflowing the bar", () => {
  const view = deriveContextView(
    {
      input_tokens: 20000,
      breakdown: { ...breakdown, messages: 18000 },
      context_length: 16000,
    },
    null,
  );
  assert.equal(view?.percent, 100);
  assert.equal(view?.free, 0);
});

test("no view without a window or a breakdown", () => {
  assert.equal(deriveContextView({ input_tokens: 10, breakdown }, null), null);
  assert.equal(
    deriveContextView({ input_tokens: 10, context_length: 100 }, null),
    null,
  );
});

test("the composer shows the report ahead of every send path and sends nothing", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const submit = thread.slice(
    thread.indexOf("const handleSubmit = useCallback("),
  );
  const start = submit.indexOf("isContextCommand(composerText)");
  assert.ok(start > 0);
  assert.ok(start < submit.indexOf("if (disabled || shouldBlockSend())"));
  assert.ok(start < submit.indexOf("interceptSend(event)"));
  const branch = submit.slice(start, submit.indexOf("showContext(", start));
  // Nothing in the branch can reach the thread, the queue or a request.
  for (const call of [
    "append(",
    "send(",
    "enqueue",
    "requestSubmit",
    "Compaction",
  ]) {
    assert.ok(!branch.includes(call), call);
  }
  // Under the messages, not one of them: it is never saved or sent.
  assert.match(
    thread,
    /<ProgressiveMessages[\s\S]*?\/>\s*<ContextReportMount \/>/,
  );
});

test("only the count carries /context's flags, never the completion", () => {
  // The completion body is the prompt cache's key: /context's flags must stay out of it.
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.ok(!adapter.includes("breakdown"));
  assert.ok(
    !readSrc("features/chat/api/context-breakdown.ts").includes(
      "/chat/completions",
    ),
  );
});

type CountModule = {
  countContext: (
    branch: readonly unknown[],
    threadId: string | undefined,
  ) => Promise<{ input_tokens: number } | null>;
};

function loadCount(
  state: Record<string, unknown>,
  counted: Record<string, unknown>,
) {
  const sent: Record<string, unknown>[] = [];
  const module = loadWithStubs<CountModule>(
    new URL("../src/features/chat/api/context-breakdown.ts", import.meta.url),
    {
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => state },
      },
      "./chat-adapter": {
        buildLocalTokenCountHistory: async () => ({
          messages: [{ role: "user", content: "hi" }],
        }),
        buildLocalTokenCountExtras: async () => ({
          enable_tools: false,
          thread_id: "t1",
        }),
        buildLocalTokenCountReasoning: () => ({}),
      },
      "./chat-api": {
        countChatInputTokens: async (payload: Record<string, unknown>) => {
          sent.push(payload);
          return counted;
        },
      },
    },
  );
  return { module, sent };
}

function runtimeState(overrides: Record<string, unknown> = {}) {
  const published: unknown[] = [];
  return {
    published,
    state: {
      activeThreadId: "t1",
      contextUsage: { totalTokens: 349 },
      params: { checkpoint: "org/A-GGUF", maxTokens: 2048 },
      runningByThreadId: {},
      setContextUsage: (usage: unknown) => published.push(usage),
      ...overrides,
    },
  };
}

test("the count is the recount's payload plus /context's own fields", async () => {
  const { state } = runtimeState();
  const { module, sent } = loadCount(state, {
    input_tokens: 348,
    model: "org/A-GGUF",
  });
  await module.countContext([], "t1");
  assert.deepEqual(sent, [
    {
      model: "org/A-GGUF",
      messages: [{ role: "user", content: "hi" }],
      enable_tools: false,
      thread_id: "t1",
      breakdown: true,
      max_tokens: 2048,
    },
  ]);
});

test("the ring takes /context's count, so the two never disagree", async () => {
  const { state, published } = runtimeState();
  const { module } = loadCount(state, {
    input_tokens: 348,
    model: "org/A-GGUF",
  });
  await module.countContext([], "t1");
  assert.deepEqual(published, [
    {
      promptTokens: 348,
      completionTokens: 0,
      totalTokens: 348,
      cachedTokens: 0,
      cacheWriteTokens: 0,
    },
  ]);
});

for (const [name, overrides, counted] of [
  ["another model answered", {}, { input_tokens: 348, model: "org/B-GGUF" }],
  [
    "another thread is showing",
    { activeThreadId: "t2" },
    { input_tokens: 348 },
  ],
  [
    "a run is going",
    { runningByThreadId: { t1: true } },
    { input_tokens: 348 },
  ],
] as const) {
  test(`the ring keeps its number when ${name}`, async () => {
    const { state, published } = runtimeState(overrides);
    const { module } = loadCount(state, counted);
    const result = await module.countContext([], "t1");
    assert.deepEqual(published, []);
    if (name === "another model answered") assert.equal(result, null);
  });
}

test("the count carries the completion's overflow policy", () => {
  // truncate_oldest puts search_conversation and the checkpoint nudge in a compacted prompt.
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const extras = adapter.slice(
    adapter.indexOf("export async function buildLocalTokenCountExtras("),
  );
  assert.match(
    extras.slice(0, extras.indexOf("if (!supportsTools)")),
    /ggufCompactionRequestFields\(\{\s*\.\.\.localCompactionEngine\(state, state\.params\.checkpoint\),\s*autoCompactEnabled: state\.autoCompactEnabled,/,
  );
});

test("/compact clears a /context report, whose numbers the reset makes stale", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /requestThreadCompaction\(referenceThreadId\);[\s\S]{0,200}useContextReportStore\.getState\(\)\.dismissContext\(\);/,
  );
});

test("the card shows only in the thread that asked, until a message lands after it", async () => {
  const { shownContextReport } = await import(
    "../src/features/chat/stores/context-report-store.ts"
  );
  const report = { threadKey: "pane-a", lastMessageId: "m2", nonce: 1 };
  assert.equal(shownContextReport(report, "pane-a", "m2"), report);
  // A turn since makes the numbers stale.
  assert.equal(shownContextReport(report, "pane-a", "m3"), null);
  // Another compare pane, or no report at all.
  assert.equal(shownContextReport(report, "pane-b", "m2"), null);
  assert.equal(shownContextReport(null, "pane-a", "m2"), null);
});

test("an API model's window comes from the catalogue entry of its saved connection", (t) => {
  const asked: unknown[] = [];
  const { externalModelContextLength } = loadWithStubs<{
    externalModelContextLength: (checkpoint: string | null) => number | null;
  }>(new URL("../src/features/chat/external-providers.ts", import.meta.url), {
    "./model-catalog.ts": {
      modelCatalogSupportsVision: () => false,
      resolveModelCatalogEntry: (providerType: string, modelId: string) => {
        asked.push([providerType, modelId]);
        return { contextLength: 200_000 };
      },
    },
    "./custom-reasoning.ts": {
      normalizeCustomReasoningConfig: (value: unknown) => value,
    },
  });
  const { storage } = installLocalStorageFake();
  storage.setItem(
    "unsloth_chat_external_providers",
    JSON.stringify([
      {
        id: "c1",
        providerType: "anthropic",
        name: "Claude",
        baseUrl: "",
        models: [],
      },
    ]),
  );
  t.after(() => {
    Reflect.deleteProperty(globalThis, "window");
  });
  Object.assign(globalThis, { window: globalThis });
  assert.equal(externalModelContextLength("external::c1::claude-x"), 200_000);
  assert.deepEqual(asked, [["anthropic", "claude-x"]]);
  // A local model, or a connection that is gone, has no catalogued window.
  assert.equal(externalModelContextLength("org/A-GGUF"), null);
  assert.equal(externalModelContextLength("external::gone::claude-x"), null);
});

test("usage is a whole percent of the window, capped at 100", () => {
  assert.equal(usagePercent(1_791, 200_000), 1);
  assert.equal(usagePercent(150_000, 200_000), 75);
  assert.equal(usagePercent(250_000, 200_000), 100);
});
