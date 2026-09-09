// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";
import type { CarriedPart as Carried } from "../src/features/chat/utils/chat-generation-recovery.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const recovery = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);
const parser = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);
const { createGenerationToolRecovery } = await import(
  "../src/features/chat/utils/generation-tool-recovery.ts"
);

const start = (id = "call_0") => ({
  type: "tool_start",
  tool_call_id: id,
  tool_name: "edit_file",
  arguments: { path: "scene.glsl" },
});
const end = (id = "call_0") => ({
  type: "tool_end",
  tool_call_id: id,
  result: "ok",
});

test("replay adds new cards and applies their results", () => {
  const carried: Carried[] = [];
  const replay = createGenerationToolRecovery(carried, "run");
  replay(start(), 12, 1);
  const pending = carried[0].part as Record<string, unknown>;
  assert.equal(pending.result, undefined);
  replay(end(), 20, 2);
  assert.equal(carried.length, 1);
  assert.equal(carried[0].at, 12);
  assert.deepEqual(carried[0].part, { ...pending, result: "ok" });
  assert.equal(
    pending.result,
    undefined,
    "published snapshots must not be mutated",
  );
});

test("replay resumes an existing card without duplicating it", () => {
  const saved = {
    type: "tool-call",
    toolCallId: "call_0:saved-id",
    toolName: "edit_file",
    args: { path: "scene.glsl" },
    argsText: '{"path":"scene.glsl"}',
  };
  const carried = [{ at: 12, part: saved }];
  createGenerationToolRecovery(carried, "run")(end(), 20, 2);
  assert.deepEqual(carried, [{ at: 12, part: { ...saved, result: "ok" } }]);
});

test("reused backend ids get separate cards across rounds and reloads", () => {
  const carried: Carried[] = [];
  let replay = createGenerationToolRecovery(carried, "run");
  replay(start(), 0, 1);
  replay(end(), 0, 2);
  replay(start(), 10, 3);
  replay = createGenerationToolRecovery(carried, "run");
  replay({ ...end(), result: "second" }, 10, 4);
  assert.equal(carried.length, 2);
  const parts = carried.map((entry) => entry.part as Record<string, unknown>);
  assert.notEqual(parts[0].toolCallId, parts[1].toolCallId);
  assert.deepEqual(
    parts.map((part) => part.result),
    ["ok", "second"],
  );
});

test("wrapped events preserve exact arguments and completion argument updates", () => {
  const carried: Carried[] = [];
  const replay = createGenerationToolRecovery(carried, "run");
  replay(
    {
      _toolEvent: {
        ...start(),
        arguments: { id: Number("9007199254740993"), path: "old" },
        arguments_text: '{"id":9007199254740993,"path":"old"}',
      },
    },
    0,
    1,
  );
  replay({ _toolEvent: { ...end(), arguments: { path: "new" } } }, 0, 2);
  assert.equal(
    (carried[0].part as Record<string, unknown>).argsText,
    '{"id":9007199254740993,"path":"new"}',
  );
});

test("sandbox results retain files, images and the run's session", () => {
  const carried: Carried[] = [];
  const replay = createGenerationToolRecovery(carried, "run");
  replay({ ...start(), tool_name: "python" }, 0, 1);
  replay(
    {
      ...end(),
      result:
        'done\n__FILES__:[{"name":"plot.png","size":7}]\n__IMAGES__:["plot.png"]',
    },
    0,
    2,
    "saved-session",
  );
  assert.deepEqual((carried[0].part as Record<string, unknown>).result, {
    text: "done",
    files: [{ name: "plot.png", size: 7 }],
    images: ["plot.png"],
    sessionId: "saved-session",
  });
});

test("MCP image results remain structured and malformed envelopes remain readable", () => {
  for (const [result, expected] of [
    [
      'done\n__MCP_IMAGES__:[{"data":"abc","mimeType":"image/png"}]',
      { text: "done", images: [{ data: "abc", mimeType: "image/png" }] },
    ],
    ["done\n__MCP_IMAGES__:invalid", "done\n__MCP_IMAGES__:invalid"],
  ]) {
    const carried: Carried[] = [];
    const replay = createGenerationToolRecovery(carried, "run");
    replay(start(), 0, 1);
    replay({ ...end(), result }, 0, 2);
    assert.deepEqual(
      (carried[0].part as Record<string, unknown>).result,
      expected,
    );
  }
});

test("unrelated events and unmatched completions cannot alter saved cards", () => {
  const carried: Carried[] = [];
  const replay = createGenerationToolRecovery(carried, "run");
  replay(end(), 0, 1);
  replay({ type: "tool_start" }, 0, 2);
  replay(null, 0, 3);
  assert.deepEqual(carried, []);
});

const provider = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  "utf8",
);
const scheduler = provider.slice(
  provider.indexOf("function scheduleGenerationRecovery("),
  provider.indexOf("\nexport async function ensureThreadRecord("),
);
const executable = ts.transpileModule(scheduler, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
  },
}).outputText;

async function recoverRun(content: unknown[], payloads: unknown[]) {
  const snapshots: Array<{
    content: Record<string, unknown>[];
    metadata: Record<string, unknown>;
  }> = [];
  const generationRecoveries = new Map();
  const runtime = {
    models: [],
    registerThreadServerCancel() {},
    setThreadRunning() {},
    clearThreadServerCancel() {},
  };
  const run = {
    id: "run",
    threadId: "thread",
    assistantMessageId: "msg",
    status: "completed",
    lastEventSeq: payloads.length,
    requestPayload: { model: "test", session_id: "saved-session" },
    createdAt: 1,
    startedAt: 1,
    completedAt: 100,
  };
  const context = vm.createContext({
    ...recovery,
    ...parser,
    createGenerationToolRecovery,
    generationRecoveries,
    useChatRuntimeStore: { getState: () => runtime },
    cancelChatGenerationRun: async () => {},
    budgetImpliesTruncation: () => false,
    saveStoredChatMessage: async (message: (typeof snapshots)[number]) => {
      snapshots.push(structuredClone(message));
    },
    followChatGenerationRun: async function* () {
      for (let i = 0; i < payloads.length; i++) {
        const update = {
          run,
          event: {
            seq: i + 1,
            type: "chunk",
            payload: payloads[i],
            createdAt: i + 1,
          },
        };
        yield update;
        yield update;
      }
    },
    isTerminalChatGenerationRun: () => true,
    forgetServerActiveGenerationRun() {},
    ChatGenerationStalledError: class extends Error {},
  });
  vm.runInContext(executable, context);
  context.scheduleGenerationRecovery(
    "thread",
    {
      id: "msg",
      content,
      createdAt: 1,
      metadata: {
        generationRunId: "run",
        generationSeq: 0,
        generationStatus: "running",
        generationSettled: false,
      },
    },
    {
      threadListItem: () => ({ getState: () => ({ remoteId: "other-view" }) }),
    },
  );
  await generationRecoveries.get("run").promise;
  const final = snapshots.at(-1);
  assert.ok(final);
  assert.equal(final.metadata.generationSettled, true);
  assert.equal(final.metadata.generationSeq, payloads.length);
  return final.content;
}

test("the recovery scheduler persists later tool events between reasoning groups", async () => {
  const content = await recoverRun(
    [],
    [
      { choices: [{ delta: { reasoning_content: "before" } }] },
      start(),
      end(),
      { choices: [{ delta: { reasoning_content: "after" } }] },
      { choices: [{ delta: { content: "done" } }] },
    ],
  );
  assert.deepEqual(
    content.map((part) => part.type),
    ["reasoning", "tool-call", "reasoning", "text"],
  );
  assert.equal(content[1].result, "ok");
  assert.deepEqual(
    content
      .filter((part) => part.type === "reasoning")
      .map((part) => part.text),
    ["before", "after"],
  );
});

test("the recovery scheduler completes a saved pending card", async () => {
  const content = await recoverRun(
    [
      {
        type: "tool-call",
        toolCallId: "call_0:saved-id",
        toolName: "python",
        args: {},
      },
    ],
    [end()],
  );
  assert.equal(content.length, 1);
  assert.equal(content[0].toolCallId, "call_0:saved-id");
  assert.deepEqual(content[0].result, {
    text: "ok",
    images: [],
    files: [],
    sessionId: "saved-session",
  });
});

test("explicit think tags do not shift replayed tool offsets", async () => {
  const content = await recoverRun(
    [],
    [
      { choices: [{ delta: { content: "<think>before</think>" } }] },
      start(),
      end(),
      { choices: [{ delta: { content: "<think>after</think>done" } }] },
    ],
  );
  assert.deepEqual(
    content.map((part) => part.type),
    ["reasoning", "tool-call", "reasoning", "text"],
  );
  assert.deepEqual(
    content
      .filter((part) => part.type === "reasoning")
      .map((part) => part.text),
    ["before", "after"],
  );
});
