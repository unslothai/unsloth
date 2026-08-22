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

test("an abandoned turn is pruned with the user prompt that triggered it", () => {
  assert.match(
    adapter,
    /isAnthropicRefusalMessage\(message\) \|\|\s*isAbandonedAssistantTurn\(message, includeReasoningContent\)/,
  );
  assert.match(adapter, /if \(last && last\.role === "user"\) surviving\.pop\(\)/);
});

test("the send and token-count paths prune through the same helper", () => {
  assert.equal(adapter.match(/pruneOutboundHistory\(/g)?.length, 3);
  assert.doesNotMatch(adapter, /survivingMessages\.push\(message\)/);
});
