// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const adapterPath = fileURLToPath(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
);
const source = readFileSync(adapterPath, "utf8");
const start = source.indexOf("const rawDeltaToolCalls = (");
const end = source.indexOf("if (!delta && !reasoning)", start);
assert.ok(start >= 0 && end > start, "OpenAI tool-call stream handler was not found");
const handler = source.slice(start, end);

test("a Codex tool delta and Studio execution events share one card id", () => {
  assert.match(
    handler,
    /const stablePartId = stableId\s*\? resolveToolPartId\(stableId\)/,
  );
  assert.match(
    handler,
    /toolCallId === stablePartId/,
  );
  assert.match(
    handler,
    /const callId =\s*stablePartId \|\| `tool_call_/,
  );
});
