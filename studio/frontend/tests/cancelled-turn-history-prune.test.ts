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
    /!only\.content &&\s*!only\.tool_calls &&\s*!only\.reasoning_content/,
  );
});

test("a turn that carries payload of its own is never abandoned", () => {
  assert.match(adapter, /function assistantTurnCarriesPayload\(/);
  assert.match(adapter, /if \(assistantTurnCarriesPayload\(message\)\) return false;/);
});

test("a turn that finished on reasoning alone keeps its prompt", () => {
  assert.match(adapter, /function assistantTurnEndedEarly\(/);
  assert.match(adapter, /!assistantTurnEndedEarly\(message\) &&/);
});

test("an abandoned turn is pruned with the user prompt that triggered it", () => {
  assert.match(adapter, /if \(last && last\.role === "user"\) surviving\.pop\(\)/);
});

test("a trailing abandoned turn keeps the prompt it followed", () => {
  assert.match(adapter, /if \(refused \|\| index < lastSurviving\) \{/);
});

test("the send and token-count paths prune through the same helper", () => {
  assert.match(
    adapter,
    /const outboundMessages = pruneOutboundHistory\(messages, true\)/,
  );
  assert.match(
    adapter,
    /const survivingMessages = pruneOutboundHistory\(\s*messages,\s*!isExternalRequest,\s*\);/,
  );
});
