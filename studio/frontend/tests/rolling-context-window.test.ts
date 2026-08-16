// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const transport = readFileSync(
  new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
  "utf8",
);

test("local chat opts into the rolling context policy", () => {
  assert.match(adapter, /isGguf === true/);
  assert.match(adapter, /context_overflow:\s*"truncate_oldest"/);
  assert.match(adapter, /Older turns omitted from model context/);
});

test("the transport preserves standard chunks with context metadata", () => {
  assert.doesNotMatch(transport, /parsed\.type === "context_truncated"/);
  assert.match(adapter, /chunk\.context_truncated/);
});
