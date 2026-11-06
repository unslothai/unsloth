// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { addCodexReasoning, codexReasoningForToolCalls, readOpenAIResponsesReasoning,
  type CodexReasoningLedger } from "../src/features/chat/codex-reasoning.ts";
import { readSrc } from "./helpers/kit.ts";

test("Responses reasoning remains bound to its tool round and replay metadata", () => {
  const item = { type: "reasoning", id: "rs_1", encrypted_content: "opaque" };
  let ledger: CodexReasoningLedger = { byToolCall: {} };
  ledger = addCodexReasoning(ledger, [item], ["call_1"]);

  const stored = readOpenAIResponsesReasoning({
    custom: { openaiResponsesReasoning: ledger },
  });
  assert.deepEqual(codexReasoningForToolCalls(stored, ["call_1"]), [item]);
  assert.equal(codexReasoningForToolCalls(stored, ["call_2"]), undefined);
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /extraRecord\.openai_responses_reasoning/);
  assert.match(adapter, /openai_responses_reasoning:\s*reasoning/);
});
