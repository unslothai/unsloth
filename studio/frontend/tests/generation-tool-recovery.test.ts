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
    snapshots,
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

test("replay persists document citations without duplicate sources", async () => {
  const citation = {
    type: "page_location",
    document_title: "Report",
    document_index: 0,
    start_page_number: 1,
    end_page_number: 2,
    cited_text: "Supporting passage",
  };
  const event = { type: "document_citations", citations: [citation] };
  const result = await recoverRun(
    [],
    [
      { choices: [{ delta: { content: "Answer [1]" } }] },
      { _toolEvent: event },
      event,
    ],
  );
  assert.equal(recovery.generationRawContent(result.content).raw, "Answer [1]");
  assert.deepEqual(
    result.content.filter((part) => part.type === "source"),
    [
      {
        type: "source",
        sourceType: "url",
        id: "#anthropic-doc-0#page_location:1:2",
        url: "#anthropic-doc-0",
        title: "Report",
        metadata: { description: "Supporting passage" },
      },
    ],
  );
  assert.deepEqual(result.shown, result.content);
  const resumed = await recoverRun(result.content, [event, event], {
    cursor: 1,
  });
  assert.equal(
    resumed.content.filter((part) => part.type === "source").length,
    1,
  );
});

test("research handoffs stay hidden and approval cards finish", async () => {
  for (const gated of [false, true]) {
    const event = {
      ...start(),
      tool_name: "deep_research",
      arguments: { question: "Research this" },
      ...(gated
        ? { awaiting_confirmation: true, approval_id: "approval" }
        : {}),
    };
    const result = await recoverRun(
      [],
      [
        event,
        {
          ...end(),
          tool_name: "deep_research",
          result: "Deep Research has started",
        },
      ],
    );
    assert.equal(result.content.length, gated ? 1 : 0);
    if (gated)
      assert.equal(result.content[0].result, "Deep Research has started");
  }
});

test("Gemini native results and later images survive reloads", async () => {
  const code = {
    executableCode: { language: "PYTHON", code: "print(1)" },
    thoughtSignature: "code-signature",
  };
  const output = {
    codeExecutionResult: { outcome: "OUTCOME_OK", output: "1" },
  };
  const image = {
    inlineData: { mimeType: "image/png", data: "cGxvdA==" },
    thoughtSignature: "image-signature",
  };
  const event = {
    ...start(),
    tool_name: "code_execution",
    arguments: { google: { native_part: { parts: [code] } } },
  };
  const completion = {
    ...end(),
    tool_name: "code_execution",
    google: { native_part: { parts: [output] } },
  };
  const saved = {
    type: "tool-call",
    toolCallId: "call_0:saved",
    backendToolCallId: "call_0",
    toolName: "code_execution",
    args: event.arguments,
  };
  const result = await recoverRun([saved], [event, completion], { cursor: 1 });
  const args = result.content[0].args as typeof event.arguments;
  assert.deepEqual(args.google.native_part.parts, [code, output]);
  const resumed = await recoverRun(
    result.content,
    [
      event,
      completion,
      {
        ...end(),
        tool_name: "code_execution",
        google: { native_part: { parts: [image] } },
        result: '1\n__IMAGES__:["plot.png"]',
      },
    ],
    { cursor: 2 },
  );
  assert.equal(resumed.content.length, 1);
  const merged = resumed.content[0].args as typeof event.arguments;
  assert.deepEqual(merged.google.native_part.parts, [code, output, image]);
  assert.deepEqual(
    JSON.parse(String(resumed.content[0].argsText)).google.native_part.parts,
    [code, output, image],
  );
  assert.deepEqual(saved.args.google.native_part.parts, [code]);
});

test("legacy Gemini signatures and image calls retain exact arguments", () => {
  const code = {
    executableCode: { code: "print(1)" },
    thought_signature: "legacy-signature",
  };
  const image = {
    inlineData: { mimeType: "image/png", data: "aW1hZ2U=" },
    thoughtSignature: "image-signature",
  };
  for (const toolName of ["code_execution", "image_generation"]) {
    const carried: Carried[] = [];
    const replay = createGenerationToolRecovery(carried, "run").apply;
    const argsText = `{"id":9007199254740993,"google":{"native_part":${JSON.stringify(toolName === "code_execution" ? code : {})}}}`;
    replay(
      {
        ...start(),
        tool_name: toolName,
        arguments: JSON.parse(argsText),
        arguments_text: argsText,
      },
      0,
      1,
    );
    replay(
      {
        ...end(),
        tool_name: toolName,
        google: { native_part: { parts: [image] } },
        ...(toolName === "image_generation"
          ? { image_b64: "aW1hZ2U=", image_mime: "image/png" }
          : {}),
      },
      0,
      2,
    );
    const part = carried[0].part as Record<string, unknown>;
    const expected =
      toolName === "code_execution"
        ? [
            {
              executableCode: code.executableCode,
              thoughtSignature: "legacy-signature",
            },
            image,
          ]
        : [image];
    assert.deepEqual(
      JSON.parse(String(part.argsText)).google.native_part.parts,
      expected,
    );
    assert.match(String(part.argsText), /9007199254740993/);
    if (toolName === "image_generation")
      assert.equal(
        (part.result as Record<string, unknown>).image_b64,
        "aW1hZ2U=",
      );
  }
});

test("citations retain safe URLs and distinct footnotes", async () => {
  const base = {
    type: "char_location",
    document_index: 2,
    start_char_index: 0,
    end_char_index: 10,
    cited_text: "x".repeat(300),
  };
  const result = await recoverRun(
    [],
    [
      {
        type: "document_citations",
        citations: [
          null,
          [],
          { ...base, source: "javascript:alert(1)" },
          { ...base, source: "https://example.com/report" },
          {
            ...base,
            source: "https://example.com/report",
            start_char_index: 10,
            end_char_index: 20,
          },
        ],
      },
    ],
  );
  assert.equal(result.content.length, 3);
  assert.deepEqual(
    result.content.map((part) => part.url),
    [
      "#anthropic-doc-2",
      "https://example.com/report",
      "https://example.com/report",
    ],
  );
  assert.equal(new Set(result.content.map((part) => part.id)).size, 3);
  assert.ok(
    result.content.every(
      (part) =>
        (part.metadata as { description: string }).description.length === 243,
    ),
  );
});

test("calls a provider gave no id to open one card each", async () => {
  const call = (name: string) => ({
    type: "tool_start",
    tool_call_id: "",
    tool_name: name,
    arguments: { path: `${name}.ts` },
  });
  const { content } = await recoverRun(
    [],
    [call("read_file"), call("edit_file")],
  );
  const cards = content.filter((part) => part.type === "tool-call");
  assert.deepEqual(
    cards.map((part) => part.toolName),
    ["read_file", "edit_file"],
  );
  assert.equal(new Set(cards.map((part) => part.toolCallId)).size, 2);
});

test("an id-less completion closes the most recent open card", async () => {
  const { content } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "", tool_name: "read_file" },
      { type: "tool_start", tool_call_id: "", tool_name: "edit_file" },
      { type: "tool_end", tool_call_id: "", result: "wrote it" },
    ],
  );
  const cards = content.filter((part) => part.type === "tool-call");
  assert.deepEqual(
    cards.map((part) => [part.toolName, part.result]),
    [
      ["read_file", undefined],
      ["edit_file", "wrote it"],
    ],
  );
});

test("saved id-less cards each keep their own pending slot", async () => {
  const saved = (name: string) => ({
    type: "tool-call",
    toolCallId: `${name}:saved`,
    backendToolCallId: "",
    toolName: name,
    args: {},
    argsText: "{}",
  });
  const { content } = await recoverRun(
    [saved("read_file"), saved("edit_file")],
    [
      { type: "tool_end", tool_call_id: "", result: "second" },
      { type: "tool_end", tool_call_id: "", result: "first" },
    ],
  );
  const cards = content.filter((part) => part.type === "tool-call");
  assert.deepEqual(
    cards.map((part) => [part.toolName, part.result]),
    [
      ["read_file", "first"],
      ["edit_file", "second"],
    ],
  );
});

test("a full identity replay does not republish the saved history", async () => {
  const legacy = {
    type: "tool-call",
    toolCallId: "call_7:legacy",
    toolName: "edit_file",
    args: {},
    argsText: "{}",
  };
  const history = Array.from({ length: 12 }, (_, i) => ({
    choices: [{ delta: { content: `t${i}` } }],
  }));
  const { replayFrom, imports } = await recoverRun(
    [legacy],
    [
      ...history,
      { type: "tool_start", tool_call_id: "call_7", tool_name: "edit_file" },
      { type: "tool_end", tool_call_id: "call_7", result: "done" },
    ],
    { cursor: 12 },
  );
  assert.equal(replayFrom, 0, "a legacy card still replays from the start");
  assert.ok(
    imports.length <= 4,
    `history must not be republished, got ${imports.length} imports`,
  );
});

test("recovery rebuilds the Sources panel from a replayed web search", async () => {
  const result = [
    "Title: Unsloth docs",
    "URL: https://docs.unsloth.ai/",
    "Snippet: fine-tuning guide",
  ].join("\n");
  const { content } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "ws_1", tool_name: "web_search" },
      { type: "tool_end", tool_call_id: "ws_1", result },
    ],
  );
  const sources = content.filter((part) => part.type === "source");
  assert.deepEqual(
    sources.map((part) => [part.url, part.title]),
    [["https://docs.unsloth.ai/", "Unsloth docs"]],
  );
  assert.equal(
    content.at(-1)?.type,
    "source",
    "sources trail the reply, as the live path yields them",
  );
});

test("rebuilt sources do not duplicate ones already saved", async () => {
  const result = "Title: Docs\nURL: https://docs.unsloth.ai/\nSnippet: guide";
  const saved = [
    {
      type: "tool-call",
      toolCallId: "ws_1:saved",
      backendToolCallId: "ws_1",
      toolName: "web_search",
      args: {},
      argsText: "{}",
      result,
    },
    {
      type: "source",
      sourceType: "url",
      id: "https://docs.unsloth.ai/",
      url: "https://docs.unsloth.ai/",
      title: "Docs",
    },
  ];
  const { content } = await recoverRun(saved, [
    { choices: [{ delta: { content: "more" } }] },
  ]);
  assert.equal(content.filter((part) => part.type === "source").length, 1);
});

test("an unsafe search URL never reaches the Sources panel", async () => {
  const { content } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "ws_2", tool_name: "web_fetch" },
      {
        type: "tool_end",
        tool_call_id: "ws_2",
        result: "Title: Bad\nURL: javascript:alert(1)\nSnippet: no",
      },
    ],
  );
  assert.equal(content.filter((part) => part.type === "source").length, 0);
});

test("a web search completed twice keeps the citation list", async () => {
  const citations = [
    "Title: Unsloth docs",
    "URL: https://docs.unsloth.ai/",
    "Snippet: guide",
  ].join("\n");
  const { content } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "ws_0", tool_name: "web_search" },
      { type: "tool_end", tool_call_id: "ws_0", result: "Searching: unsloth" },
      { type: "tool_end", tool_call_id: "ws_0", result: citations },
    ],
  );
  const card = content.find((part) => part.type === "tool-call");
  assert.equal(card?.result, citations);
  assert.deepEqual(
    content.filter((part) => part.type === "source").map((part) => part.url),
    ["https://docs.unsloth.ai/"],
  );
});

test("a start on the same id still opens a second round", async () => {
  const { content } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "call_0", tool_name: "edit_file" },
      { type: "tool_end", tool_call_id: "call_0", result: "first" },
      { type: "tool_start", tool_call_id: "call_0", tool_name: "edit_file" },
      { type: "tool_end", tool_call_id: "call_0", result: "second" },
    ],
  );
  assert.deepEqual(
    content
      .filter((part) => part.type === "tool-call")
      .map((part) => part.result),
    ["first", "second"],
  );
});

test("a legacy id-less pending card is matched through replay", async () => {
  const legacy = {
    type: "tool-call",
    toolCallId: "edit_file_1757000000000",
    toolName: "edit_file",
    args: {},
    argsText: "{}",
  };
  const { content, replayFrom } = await recoverRun(
    [legacy],
    [
      { type: "tool_start", tool_call_id: "", tool_name: "edit_file" },
      { type: "tool_end", tool_call_id: "", result: "done" },
    ],
    { cursor: 1 },
  );
  assert.equal(replayFrom, 0, "an identity-less card replays from the start");
  const cards = content.filter((part) => part.type === "tool-call");
  assert.equal(
    cards.length,
    1,
    "the historical start must not open a second card",
  );
  assert.equal(cards[0].result, "done");
});

test("a citation result reaches a card whose placeholder was already saved", async () => {
  const citations = [
    "Title: Unsloth docs",
    "URL: https://docs.unsloth.ai/",
    "Snippet: guide",
  ].join("\n");
  const saved = {
    type: "tool-call",
    toolCallId: "ws_0:saved",
    backendToolCallId: "ws_0",
    toolName: "web_search",
    args: {},
    argsText: "{}",
    result: "Searching: unsloth",
  };
  const { content } = await recoverRun(
    [saved],
    [{ type: "tool_end", tool_call_id: "ws_0", result: citations }],
  );
  const cards = content.filter((part) => part.type === "tool-call");
  assert.equal(cards.length, 1);
  assert.equal(cards[0].result, citations);
  assert.deepEqual(
    content.filter((part) => part.type === "source").map((part) => part.url),
    ["https://docs.unsloth.ai/"],
  );
});

test("a saved completed card still yields to a new round on its id", async () => {
  const saved = {
    type: "tool-call",
    toolCallId: "call_0:saved",
    backendToolCallId: "call_0",
    toolName: "edit_file",
    args: {},
    argsText: "{}",
    result: "first",
  };
  const { content } = await recoverRun(
    [saved],
    [
      { type: "tool_start", tool_call_id: "call_0", tool_name: "edit_file" },
      { type: "tool_end", tool_call_id: "call_0", result: "second" },
    ],
  );
  assert.deepEqual(
    content
      .filter((part) => part.type === "tool-call")
      .map((part) => part.result),
    ["first", "second"],
  );
});

test("a settled recovery reports the tool calls it restored", async () => {
  const { content, snapshots } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "call_0", tool_name: "edit_file" },
      { type: "tool_end", tool_call_id: "call_0", result: "ok" },
    ],
  );
  assert.equal(content.filter((part) => part.type === "tool-call").length, 1);
  const final = snapshots.at(-1);
  const details = final?.metadata.responseDetails as { toolCalls: string[] };
  const timing = final?.metadata.timing as { toolCallCount: number };
  assert.deepEqual(details.toolCalls, ["edit_file"]);
  assert.equal(timing.toolCallCount, 1);
});

test("repeated source ids pair one to one instead of multiplying", () => {
  const url = "https://docs.unsloth.ai/";
  const source = () => ({ type: "source", sourceType: "url", id: url, url });
  const view = [{ type: "text", text: "answer" }, source(), source()];
  const recovered = [{ type: "text", text: "answer" }, source(), source()];
  let carriedView = view;
  // Every publish re-imports; a duplicate that reads as missing would grow each round.
  for (let round = 0; round < 3; round++) {
    carriedView = recovery.recoveredContentToImport(
      carriedView,
      recovered,
    ) as typeof view;
    assert.equal(
      carriedView.filter((part) => part.type === "source").length,
      2,
      `round ${round}`,
    );
  }
});

test("a legacy completed card takes its citation result", async () => {
  const citations = "Title: Docs\nURL: https://docs.unsloth.ai/\nSnippet: g";
  const legacy = {
    type: "tool-call",
    toolCallId: "ws_0:9f1c-legacy-uuid",
    toolName: "web_search",
    args: {},
    argsText: "{}",
    result: "Searching: unsloth",
  };
  const { content } = await recoverRun(
    [legacy],
    [{ type: "tool_end", tool_call_id: "ws_0", result: citations }],
  );
  const cards = content.filter((part) => part.type === "tool-call");
  assert.equal(cards.length, 1);
  assert.equal(cards[0].result, citations);
  assert.deepEqual(
    content.filter((part) => part.type === "source").map((part) => part.url),
    ["https://docs.unsloth.ai/"],
  );
});

test("a repeated id-less ending replaces the card it already finished", async () => {
  const { content } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "", tool_name: "web_search" },
      { type: "tool_end", tool_call_id: "", result: "Searching: unsloth" },
      { type: "tool_end", tool_call_id: "", result: "final answer" },
    ],
  );
  const cards = content.filter((part) => part.type === "tool-call");
  assert.equal(cards.length, 1);
  assert.equal(cards[0].result, "final answer");
});

test("a divergent reply wins over a view that is only a tool card", () => {
  const view: Record<string, unknown>[] = [
    {
      type: "tool-call",
      toolCallId: "old:card",
      toolName: "edit_file",
      result: "stale",
    },
  ];
  const recovered: Record<string, unknown>[] = [
    { type: "text", text: "server repaired response" },
  ];
  assert.deepEqual(
    recovery.recoveredContentToImport(view, recovered),
    recovered,
  );
});

test("the id-less sentinel keeps the file searchable and cannot collide", () => {
  const src = readFileSync(
    new URL(
      "../src/features/chat/utils/generation-tool-recovery.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.ok(!src.includes("\0"), "a NUL byte makes the file read as binary");
  // A backend tool call id must satisfy ^[a-zA-Z0-9_-]+$, so '#' can never appear in one.
  for (const key of src.matchAll(/`(#idless:[^`]*)`/g)) {
    assert.ok(/^#/.test(key[1]), key[1]);
  }
  assert.equal(src.match(/`#idless:/g)?.length, 3);
});

test("a source an earlier recovery saved stays behind text replayed after it", async () => {
  const citations = "Title: Docs\nURL: https://docs.unsloth.ai/\nSnippet: g";
  // What a previous recovery session persisted: the card, then the source it appended.
  const saved = [
    { type: "text", text: "Searching." },
    {
      type: "tool-call",
      toolCallId: "ws_0:run:1",
      backendToolCallId: "ws_0",
      toolName: "web_search",
      args: {},
      argsText: "{}",
      result: citations,
    },
    {
      type: "source",
      sourceType: "url",
      id: "https://docs.unsloth.ai/",
      url: "https://docs.unsloth.ai/",
      title: "Docs",
    },
  ];
  const { content } = await recoverRun(saved, [
    { choices: [{ delta: { content: " Here is **what" } }] },
    { choices: [{ delta: { content: " I found** next." } }] },
  ]);

  // Carried at its old offset the source would split the reply, and the emphasis with it.
  assert.equal(content.at(-1)?.type, "source");
  assert.deepEqual(
    content.filter((part) => part.type === "source").map((part) => part.url),
    ["https://docs.unsloth.ai/"],
  );
  assert.equal(
    content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join(""),
    "Searching. Here is **what I found** next.",
  );
  assert.equal(
    content.filter((part) => part.type === "text").length,
    2,
    "the text either side of the card, not cut again by the source",
  );
});

test("two rounds finding the same page keep a source each, with their own titles", async () => {
  const first = "Title: Docs v1\nURL: https://docs.unsloth.ai/\nSnippet: first";
  const second = "Title: Docs v2\nURL: https://docs.unsloth.ai/\nSnippet: second";
  const { content } = await recoverRun(
    [],
    [
      { type: "tool_start", tool_call_id: "ws_0", tool_name: "web_search" },
      { type: "tool_end", tool_call_id: "ws_0", result: first },
      { type: "tool_start", tool_call_id: "ws_1", tool_name: "web_search" },
      { type: "tool_end", tool_call_id: "ws_1", result: second },
    ],
  );
  // The live path flat-maps the cards, so the panel lists the page once per round.
  assert.deepEqual(
    content.filter((part) => part.type === "source").map((part) => part.title),
    ["Docs v1", "Docs v2"],
  );
});

test("a card that finishes twice rebuilds its sources from the newer result", () => {
  // Sources are parsed once per card rather than once per rebuild, so the cache key has to
  // move whenever the result does. Keying it on the card's backend id instead, which reads
  // as the obvious choice, breaks exactly here: OpenAI Responses ends a web search twice,
  // a placeholder and then the citations, and the panel would keep the placeholder's.
  const carried: Carried[] = [];
  const { apply, withSources } = createGenerationToolRecovery(carried, "run");
  apply({ type: "tool_start", tool_call_id: "ws_0", tool_name: "web_search" }, 0, 1);
  apply(
    {
      type: "tool_end",
      tool_call_id: "ws_0",
      result: "Title: Placeholder\nURL: https://example.com/a\nSnippet: first",
    },
    0,
    2,
  );
  assert.deepEqual(
    withSources<Record<string, unknown>>([]).map((part) => part.title),
    ["Placeholder"],
  );
  apply(
    {
      type: "tool_end",
      tool_call_id: "ws_0",
      result: "Title: Cited\nURL: https://example.com/b\nSnippet: second",
    },
    0,
    3,
  );
  assert.deepEqual(
    withSources<Record<string, unknown>>([]).map((part) => part.title),
    ["Cited"],
    "the second completion's sources, not the ones parsed for the first",
  );
});

test("each rebuild yields its own source objects", () => {
  // Same reason: the parse is shared between rebuilds, the objects must not be. A caller
  // that edits a part it was handed would otherwise change what the next rebuild returns.
  const carried: Carried[] = [];
  const { apply, withSources } = createGenerationToolRecovery(carried, "run");
  apply({ type: "tool_start", tool_call_id: "ws_0", tool_name: "web_search" }, 0, 1);
  apply(
    {
      type: "tool_end",
      tool_call_id: "ws_0",
      result: "Title: Docs\nURL: https://example.com/a\nSnippet: only",
    },
    0,
    2,
  );
  const first = withSources<Record<string, unknown>>([])[0];
  first.title = "edited";
  (first.metadata as Record<string, unknown>).description = "edited";
  const second = withSources<Record<string, unknown>>([])[0];
  assert.notEqual(first, second);
  assert.equal(second.title, "Docs");
  assert.deepEqual(second.metadata, { description: "only" });
});
