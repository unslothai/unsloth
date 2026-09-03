// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Asserted against the source like the other chat-adapter tests: importing the
// module would drag in the stores and the toast layer for one closure.
const source = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

test("an auto-load that fell back to CPU warns instead of claiming plain success", () => {
  const start = source.indexOf("const showAutoLoadSuccess = (");
  assert.ok(start >= 0, "showAutoLoadSuccess is no longer defined");
  const helper = source.slice(start, source.indexOf("\n  };", start));
  assert.match(helper, /cpuFallbackReason \? toast\.warning : toast\.success/);
  assert.match(helper, /GPU acceleration is disabled for this model session/);
});

test("every auto-load path forwards the load response's fallback reason", () => {
  const calls = source.match(/\n\s*showAutoLoadSuccess\([\s\S]*?\);/g) ?? [];
  assert.ok(calls.length > 0, "no showAutoLoadSuccess call sites found");
  for (const call of calls) {
    assert.match(call, /cpu_fallback_reason/);
  }
});
