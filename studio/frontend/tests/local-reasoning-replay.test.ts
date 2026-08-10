// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const adapterPath = fileURLToPath(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
);
const source = readFileSync(adapterPath, "utf8");
const start = source.indexOf("function serializeAssistantReplayMessages(");
const end = source.indexOf("\nfunction toOpenAIMessages(", start);
assert.ok(
  start >= 0 && end > start,
  "assistant replay serializer was not found",
);
const declaration = source.slice(start, end);

const serializeAssistantReplayMessages = new Function(
  "isAnthropicRefusalMessage",
  "collectImageParts",
  "sanitizeAssistantReplayText",
  "buildReplayContent",
  "serializeAssistantToolCallPart",
  "serializeToolResultPart",
  "canReplayToolCallWithoutRoleTool",
  "shouldFlushCompletedLocalToolPair",
  "attachAssistantThoughtSignature",
  "collectAssistantTextThoughtSignature",
  "readIncompleteInfo",
  `${
    ts.transpileModule(declaration, {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    }).outputText
  }; return serializeAssistantReplayMessages;`,
)(
  () => false,
  () => [],
  (text: string) => text,
  (text: string) => text,
  (part: {
    type?: string;
    toolCallId?: string;
    toolName?: string;
    args?: unknown;
  }) =>
    part.type === "tool-call"
      ? {
          id: part.toolCallId,
          type: "function",
          function: {
            name: part.toolName,
            arguments: JSON.stringify(part.args ?? {}),
          },
        }
      : null,
  (part: {
    type?: string;
    toolCallId?: string;
    toolName?: string;
    result?: unknown;
  }) =>
    part.type === "tool-call" && part.result !== undefined
      ? {
          role: "tool",
          content: String(part.result),
          tool_call_id: part.toolCallId,
          name: part.toolName,
        }
      : null,
  () => false,
  (part: { type?: string; result?: unknown }) =>
    part.type === "tool-call" && part.result !== undefined,
  () => undefined,
  () => undefined,
  (metadata: unknown) => {
    const custom = (
      metadata as { custom?: Record<string, unknown> } | undefined
    )?.custom;
    return custom?.incomplete ?? null;
  },
) as (
  message: {
    role: "assistant";
    status?: { type: string; reason?: string };
    metadata?: { custom?: Record<string, unknown> };
    content: Array<{
      type: string;
      text?: string;
      toolCallId?: string;
      toolName?: string;
      args?: unknown;
      result?: unknown;
    }>;
  },
  includeReasoningContent?: boolean,
) => Array<Record<string, unknown>>;

const assistantTurn = {
  role: "assistant" as const,
  content: [
    { type: "reasoning", text: "First inspect the premise." },
    { type: "reasoning", text: "Then answer clearly." },
    { type: "text", text: "Here is the answer." },
  ],
};

test("local replay preserves structured assistant reasoning", () => {
  assert.deepEqual(serializeAssistantReplayMessages(assistantTurn, true), [
    {
      role: "assistant",
      content: "Here is the answer.",
      reasoning_content: "First inspect the premise.\nThen answer clearly.",
    },
  ]);
});

test("external replay does not add the local reasoning extension", () => {
  assert.deepEqual(serializeAssistantReplayMessages(assistantTurn, false), [
    { role: "assistant", content: "Here is the answer." },
  ]);
});

test("a completed reasoning-only turn preserves its answer", () => {
  assert.deepEqual(
    serializeAssistantReplayMessages(
      {
        role: "assistant",
        status: { type: "complete" },
        content: [{ type: "reasoning", text: "The answer is forty-two." }],
      },
      true,
    ),
    [
      {
        role: "assistant",
        content: "",
        reasoning_content: "The answer is forty-two.",
      },
    ],
  );
});

test("an incomplete reasoning-only turn remains an empty sentinel", () => {
  assert.deepEqual(
    serializeAssistantReplayMessages(
      {
        role: "assistant",
        status: { type: "incomplete", reason: "cancelled" },
        content: [{ type: "reasoning", text: "An unfinished thought." }],
      },
      true,
    ),
    [{ role: "assistant", content: "" }],
  );
});

test("a rehydrated incomplete reasoning-only turn remains an empty sentinel", () => {
  assert.deepEqual(
    serializeAssistantReplayMessages(
      {
        role: "assistant",
        status: { type: "complete" },
        metadata: {
          custom: { incomplete: { reason: "cancelled" } },
        },
        content: [{ type: "reasoning", text: "An unfinished thought." }],
      },
      true,
    ),
    [{ role: "assistant", content: "" }],
  );
});

test("reasoning stays with the assistant segment around a local tool call", () => {
  assert.deepEqual(
    serializeAssistantReplayMessages(
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "I should inspect the file." },
          {
            type: "tool-call",
            toolCallId: "call-1",
            toolName: "read_file",
            args: { path: "notes.txt" },
            result: "contents",
          },
          { type: "reasoning", text: "The file answers the question." },
          { type: "text", text: "Here is the answer." },
        ],
      },
      true,
    ),
    [
      {
        role: "assistant",
        content: null,
        reasoning_content: "I should inspect the file.",
        tool_calls: [
          {
            id: "call-1",
            type: "function",
            function: {
              name: "read_file",
              arguments: '{"path":"notes.txt"}',
            },
          },
        ],
      },
      {
        role: "tool",
        content: "contents",
        tool_call_id: "call-1",
        name: "read_file",
      },
      {
        role: "assistant",
        content: "Here is the answer.",
        reasoning_content: "The file answers the question.",
      },
    ],
  );
});
