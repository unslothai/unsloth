// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  addCodexReasoning,
  codexReasoningForToolCalls,
  readCodexReasoning,

  type CodexReasoningLedger,
} from "../src/features/chat/codex-reasoning.ts";

test("Codex reasoning stays with each tool round and the final answer", () => {
  let ledger: CodexReasoningLedger = { byToolCall: {} };
  ledger = addCodexReasoning(ledger, ["round-one"], ["call-1"]);
  ledger = addCodexReasoning(ledger, ["round-two"], ["call-2"]);
  ledger = addCodexReasoning(ledger, ["final"], []);

  assert.deepEqual(codexReasoningForToolCalls(ledger, ["call-1"]), ["round-one"]);
  assert.deepEqual(codexReasoningForToolCalls(ledger, ["call-2"]), ["round-two"]);
  assert.deepEqual(ledger.final, ["final"]);
});

test("legacy reasoning metadata remains replayable", () => {
  assert.deepEqual(
    readCodexReasoning({ custom: { openaiCodexReasoning: ["legacy"] } }),
    { byToolCall: {}, final: ["legacy"] },
  );
});
