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
assert.ok(start >= 0 && end > start, "assistant replay serializer was not found");
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
  `${ts.transpileModule(declaration, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText}; return serializeAssistantReplayMessages;`,
)(
  () => false,
  () => [],
  (text: string) => text,
  (text: string) => text,
  () => null,
  () => null,
  () => false,
  () => false,
  () => undefined,
  () => undefined,
) as (
  message: { role: "assistant"; content: Array<{ type: string; text: string }> },
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
