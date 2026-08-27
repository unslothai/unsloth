// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("a Stop with no output is recognised by what it puts on the wire", () => {
  assert.match(adapter, /function isAbandonedAssistantTurn\(/);
  assert.match(
    adapter,
    /!hasReplayContent\(only\.content\) &&\s*!only\.tool_calls &&\s*!only\.reasoning_content/,
  );
});

test("whitespace is not content, the way the backend already reads it", () => {
  assert.match(adapter, /function hasReplayContent\(/);
  assert.match(
    adapter,
    /if \(typeof content === "string"\) return content\.trim\(\)\.length > 0;/,
  );
  assert.match(
    adapter,
    /part\.type === "text" && part\.text\.trim\(\)\.length > 0/,
  );
});

test("a turn that carries payload of its own is never abandoned", () => {
  assert.match(adapter, /function assistantTurnCarriesPayload\(/);
  assert.match(
    adapter,
    /if \(assistantTurnCarriesPayload\(message\)\) return false;/,
  );
});

test("a tool call is left to the wire shape, not counted as payload up front", () => {
  const start = adapter.indexOf("function assistantTurnCarriesPayload(");
  const end = adapter.indexOf("function hasReplayContent(");
  assert.ok(start > 0 && end > start);
  // A call the replay can carry already leaves tool_calls on the wire; one it cannot carry
  // reaches the provider as nothing; short-circuiting here would keep the empty turn the
  // backend drops, stranding the pair this prune exists to repair.
  assert.doesNotMatch(adapter.slice(start, end), /part\.type === "tool-call"/);
});

test("a turn that finished on reasoning alone keeps its prompt", () => {
  assert.match(adapter, /function assistantTurnEndedEarly\(/);
  assert.match(adapter, /!assistantTurnEndedEarly\(message\) &&/);
});

test("an abandoned turn is pruned with the user prompt that triggered it", () => {
  assert.match(
    adapter,
    /if \(last && last\.role === "user"\) surviving\.pop\(\)/,
  );
});

test("a trailing abandoned turn keeps the prompt it followed", () => {
  assert.match(adapter, /if \(refused \|\| index < lastSurviving\) \{/);
});

test("the send and token-count paths prune through the same helper", () => {
  assert.match(
    adapter,
    /const survivingMessages = pruneOutboundHistory\(messages, true\)/,
  );
  assert.match(
    adapter,
    /const survivingMessages = pruneOutboundHistory\(\s*messages,\s*!isExternalRequest,\s*\);/,
  );
});
