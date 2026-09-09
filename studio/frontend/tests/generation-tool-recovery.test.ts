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
  const replay = createGenerationToolRecovery(carried, "run").apply;
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
  createGenerationToolRecovery(carried, "run").apply(end(), 20, 2);
  assert.deepEqual(carried, [{ at: 12, part: { ...saved, result: "ok" } }]);
});

test("reused backend ids get separate cards across rounds and reloads", () => {
  const carried: Carried[] = [];
  let replay = createGenerationToolRecovery(carried, "run").apply;
  replay(start(), 0, 1);
  replay(end(), 0, 2);
  replay(start(), 10, 3);
  replay = createGenerationToolRecovery(carried, "run").apply;
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
  const replay = createGenerationToolRecovery(carried, "run").apply;
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
  const replay = createGenerationToolRecovery(carried, "run").apply;
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
    const replay = createGenerationToolRecovery(carried, "run").apply;
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
  const replay = createGenerationToolRecovery(carried, "run").apply;
  replay(end(), 0, 1);
  replay({ type: "tool_start" }, 0, 2);
  replay(null, 0, 3);
  assert.deepEqual(carried, []);
});

test("explicit backend ids retain colons when a saved card completes", () => {
  const carried = [
    {
      at: 0,
      part: {
        type: "tool-call",
        toolCallId: "session:thread:approval",
        backendToolCallId: "provider:call_0",
        toolName: "edit_file",
        args: {},
      },
    },
  ];
  const replay = createGenerationToolRecovery(carried, "run", 3);
  assert.equal(replay.replayFrom, 3);
  replay.apply(end("provider:call_0"), 0, 4);
  assert.equal((carried[0].part as Record<string, unknown>).result, "ok");
});

test("approval history distinguishes pending calls with the same tool name", () => {
  const carried: Carried[] = ["one", "two"].map((id) => ({
    at: 0,
    part: {
      type: "tool-call",
      toolCallId: `session:thread:${id}`,
      toolName: "edit_file",
      args: {},
    },
  }));
  const replay = createGenerationToolRecovery(carried, "run", 2);
  assert.equal(replay.replayFrom, 0);
  replay.apply({ ...start("call_0"), approval_id: "one" }, 0, 1);
  replay.apply({ ...start("call_1"), approval_id: "two" }, 0, 2);
  replay.apply({ ...end("call_1"), result: "second" }, 0, 3);
  replay.apply({ ...end("call_0"), result: "first" }, 0, 4);
  assert.deepEqual(
    carried.map(({ part }) => (part as Record<string, unknown>).result),
    ["first", "second"],
  );
});

test("an unmatched completion cannot guess an approval card by tool name", () => {
  const saved = {
    type: "tool-call",
    toolCallId: "session:thread:approval",
    toolName: "edit_file",
  };
  const carried = [{ at: 0, part: saved }];
  createGenerationToolRecovery(carried, "run").apply(
    { ...end(), tool_name: "edit_file" },
    0,
    1,
  );
  assert.equal(carried[0].part, saved);
});

test("the live adapter saves identities for ordinary, approval and provider cards", () => {
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const startAt = adapter.indexOf('if (toolEvent.type === "tool_start") {');
  const endAt = adapter.indexOf(
    '} else if (toolEvent.type === "tool_end") {',
    startAt,
  );
  assert.ok(startAt >= 0 && endAt > startAt);
  const source = `function receive(toolEvent) { ${adapter.slice(startAt, endAt)} } }`;
  const compiled = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  for (const generationRunId of [null, "run"]) {
    for (const kind of ["ordinary", "approval", "provider"]) {
      const parts: Record<string, unknown>[] =
        kind === "provider"
          ? [
              {
                type: "tool-call",
                toolCallId: "provider-id",
                toolName: "edit_file",
                args: {},
              },
            ]
          : [];
      const confirmed: string[] = [];
      const context = vm.createContext({
        toolCallParts: parts,
        toolPartIdByBackendId: new Map(
          kind === "provider" ? [["call_0", "provider-id"]] : [],
        ),
        toolConfirmationIdsByBackendId: new Map(),
        toolConfirmationScopeId: "session:thread",
        sandboxSessionId: "session",
        generationRunId,
        generationSeq: 7,
        cumulativeText: "",
        toolProvenance: undefined,
        resolveToolPartId: () =>
          kind === "provider" ? "provider-id" : "call_0:live-id",
        scopedToolOutputKey: (id: string) => id,
        toolCallArgumentsText: (_text: unknown, args: unknown) =>
          JSON.stringify(args),
        mergeToolProvenance: () => undefined,
        useChatRuntimeStore: {
          getState: () => ({
            clearToolLiveOutput() {},
            clearToolFullOutput() {},
            setToolConfirmation: (id: string) => confirmed.push(id),
          }),
        },
      });
      vm.runInContext(compiled, context);
      const event = {
        ...start(),
        ...(kind === "approval"
          ? { awaiting_confirmation: true, approval_id: "approval-1" }
          : {}),
      };
      context.receive(event);
      assert.equal(parts.length, 1);
      const saved = parts[0];
      assert.equal(saved.backendToolCallId, "call_0");
      assert.equal(
        saved.generationToolCallId,
        generationRunId ? "run:7" : undefined,
      );
      assert.equal(
        saved.toolCallId,
        kind === "approval"
          ? "session:thread:approval-1"
          : kind === "provider"
            ? "provider-id"
            : "call_0:live-id",
      );
      assert.equal(
        saved.toolApprovalId,
        kind === "approval" ? "approval-1" : undefined,
      );
      assert.deepEqual(
        confirmed,
        kind === "approval" ? [saved.toolCallId] : [],
      );
      const carried = [{ at: 0, part: saved }];
      const replay = createGenerationToolRecovery(carried, "run", 7);
      assert.equal(replay.replayFrom, 7);
      replay.apply(end(), 0, 8);
      assert.equal(carried[0].part.result, "ok");
    }
  }
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

async function recoverRun(
  content: unknown[],
  payloads: unknown[],
  options: { cursor?: number; viewContent?: unknown[] } = {},
) {
  let shown = {
    messages: [
      { message: { id: "msg", content: options.viewContent ?? content } },
    ],
  };
  const imports: unknown[][] = [];
  let replayFrom: number | undefined;
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
    followChatGenerationRun: async function* (
      _id: string,
      followOptions: { replayFrom: number },
    ) {
      replayFrom = followOptions.replayFrom;
      for (let i = followOptions.replayFrom; i < payloads.length; i++) {
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
    restoredAssistantStatus: () => ({ type: "complete", reason: "stop" }),
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
        generationSeq: options.cursor ?? 0,
        generationStatus: "running",
        generationSettled: false,
      },
    },
    {
      threadListItem: () => ({ getState: () => ({ remoteId: "thread" }) }),
      thread: () => ({
        export: () => shown,
        import: (value: typeof shown) => {
          shown = value;
          imports.push(value.messages[0].message.content);
        },
      }),
    },
  );
  await generationRecoveries.get("run").promise;
  const final = snapshots.at(-1);
  assert.ok(final);
  assert.equal(final.metadata.generationSettled, true);
  assert.equal(final.metadata.generationSeq, payloads.length);
  return {
    content: final.content,
    shown: shown.messages[0].message.content,
    imports,
    replayFrom,
  };
}

test("the recovery scheduler persists later tool events between reasoning groups", async () => {
  const { content } = await recoverRun(
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
  const { content } = await recoverRun(
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
  const { content } = await recoverRun(
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

test("the scheduler recovers old approval identities without replaying saved text", async () => {
  const saved = {
    type: "tool-call",
    toolCallId: "session:thread:approval-1",
    toolName: "edit_file",
    args: { path: "scene.glsl" },
  };
  const result = await recoverRun(
    [{ type: "text", text: "Working:" }, saved],
    [
      { choices: [{ delta: { content: "Working:" } }] },
      { ...start(), approval_id: "approval-1", awaiting_confirmation: true },
      end(),
    ],
    { cursor: 2 },
  );
  assert.equal(result.replayFrom, 0);
  assert.equal(recovery.generationRawContent(result.content).raw, "Working:");
  assert.equal(result.content.length, 2);
  assert.equal(result.content[1].toolCallId, saved.toolCallId);
  assert.equal(result.content[1].result, "ok");
  assert.deepEqual(result.shown, result.content);
});

test("the scheduler imports each live call once across reused backend ids", async () => {
  for (const approval of [false, true]) {
    const viewContent = [0, 1].map((i) => ({
      type: "tool-call",
      toolCallId: approval
        ? `session:thread:approval-${i}`
        : `call_0:live-${i}`,
      toolName: "edit_file",
      args: { path: "scene.glsl" },
      result: `result-${i}`,
    }));
    const result = await recoverRun(
      [],
      [
        { ...start(), ...(approval ? { approval_id: "approval-0" } : {}) },
        { ...end(), result: "result-0" },
        { ...start(), ...(approval ? { approval_id: "approval-1" } : {}) },
        { ...end(), result: "result-1" },
      ],
      { viewContent },
    );
    assert.equal(result.content.length, 2);
    assert.deepEqual(result.shown, result.content);
    assert.deepEqual(
      result.content.map((part) => part.result),
      ["result-0", "result-1"],
    );
    assert.ok(result.imports.length > 0);
    assert.ok(result.imports.every((parts) => parts.length === 2));
  }
});
