// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  hasOnlyStudioOwnedToolHistory,
  studioToolHistoryRequestFields,
} from "../src/features/chat/utils/studio-tool-history.ts";

function assistantToolCall(provenance?: unknown) {
  return {
    role: "assistant",
    content: [
      {
        type: "tool-call",
        toolName: "python",
        toolCallId: "call_1",
        args: { code: "print(1)" },
        result: "1",
        ...(provenance === undefined ? {} : { provenance }),
      },
    ],
  };
}

test("local tool provenance marks studio-owned history", () => {
  assert.equal(
    hasOnlyStudioOwnedToolHistory([
      { role: "user", content: [{ type: "text", text: "run python" }] },
      assistantToolCall({ source: "local" }),
    ]),
    true,
  );
});

test("mixed or unmarked tool history remains a client contract", () => {
  assert.equal(
    hasOnlyStudioOwnedToolHistory([
      assistantToolCall({ source: "local" }),
      assistantToolCall({ source: "external" }),
    ]),
    false,
  );
  assert.equal(hasOnlyStudioOwnedToolHistory([assistantToolCall()]), false);
  assert.equal(
    hasOnlyStudioOwnedToolHistory([
      { role: "assistant", content: [{ type: "text", text: "plain reply" }] },
    ]),
    false,
  );
});

test("only studio-owned tool history emits the request marker", () => {
  assert.deepEqual(
    studioToolHistoryRequestFields([assistantToolCall({ source: "local" })]),
    { studio_tool_history: true },
  );
  assert.deepEqual(
    studioToolHistoryRequestFields([
      assistantToolCall({ source: "local" }),
      assistantToolCall({ source: "external" }),
    ]),
    {},
  );
});

function hostedServerSearchCall() {
  return {
    role: "assistant",
    content: [
      {
        type: "tool-call",
        toolName: "web_search",
        toolCallId: "call_search",
        args: { _server_tool: true, query: "news" },
        result: { text: "hits" },
        provenance: { source: "external" },
      },
    ],
  };
}

test("ownership ignores hosted builtins that OpenAI replay drops", () => {
  const mixed = [
    hostedServerSearchCall(),
    assistantToolCall({ source: "local" }),
  ];
  assert.deepEqual(studioToolHistoryRequestFields(mixed), {});
  assert.deepEqual(
    studioToolHistoryRequestFields(mixed, {
      toolCallSurvives: (part) => {
        const args = part.args;
        return !(
          args &&
          typeof args === "object" &&
          !Array.isArray(args) &&
          (args as { _server_tool?: unknown })._server_tool === true
        );
      },
    }),
    { studio_tool_history: true },
  );
});

test("the live request and token-count paths mark ownership after replay", () => {
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );
  const helper = adapter.indexOf(
    "function studioToolHistoryRequestFieldsAfterReplay",
  );
  const surviveFn = adapter.indexOf(
    "function toolCallPartSurvivesOpenAIReplay",
  );
  const serializeGate = adapter.indexOf(
    "serializeAssistantToolCallPart",
    surviveFn,
  );
  const first = adapter.indexOf("...studioToolHistoryRequestFieldsAfterReplay(");
  const second = adapter.indexOf(
    "...studioToolHistoryRequestFieldsAfterReplay(",
    first + 1,
  );
  assert.ok(helper >= 0);
  assert.ok(surviveFn >= 0 && surviveFn < helper);
  assert.ok(serializeGate > surviveFn);
  assert.ok(first >= 0 && second > first);
  assert.equal(
    adapter.indexOf("...studioToolHistoryRequestFields(survivingMessages)"),
    -1,
  );
});

test("the context recount forwards the tool-history request fields", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/utils/refresh-context-usage.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const historyBuild = source.indexOf(
    "const countHistory = await buildLocalTokenCountHistory(",
  );
  const countRequest = source.indexOf("await countChatInputTokens({", historyBuild);
  const historySpread = source.indexOf("...countHistory", countRequest);
  assert.ok(historyBuild >= 0 && countRequest > historyBuild);
  assert.ok(historySpread > countRequest);
});
