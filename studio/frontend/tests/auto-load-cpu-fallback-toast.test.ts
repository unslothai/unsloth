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
  // Not the exact ternary: a second warning condition (a partial GPU offload) was
  // added alongside it, and pinning the whole expression made a widening of the
  // rule read as a break of it. What matters is that a CPU fallback still selects
  // the warning toast and still explains itself.
  assert.match(helper, /cpuFallbackReason[^\n]*\? toast\.warning : toast\.success/);
  assert.match(helper, /GPU acceleration is disabled for this model session/);
});

test("an auto-load that only partly reached the GPU warns too", () => {
  // The path a new user hits first: a chat sent with no model loaded auto-loads
  // the default, so without this the case the warning exists for is the one case
  // that stays silent.
  const start = source.indexOf("const showAutoLoadSuccess = (");
  const helper = source.slice(start, source.indexOf("\n  };", start));
  assert.match(helper, /isPartialOffload/);
  assert.match(helper, /partialOffloadDescription/);
});

test("every auto-load path forwards the offload counts", () => {
  const calls = source.match(/\n\s*showAutoLoadSuccess\([\s\S]*?\);/g) ?? [];
  assert.ok(calls.length > 0, "no showAutoLoadSuccess call sites found");
  for (const call of calls) {
    assert.match(call, /offloaded_layers/);
    assert.match(call, /offload_total_layers/);
    // Manual mode pins the split deliberately, so the mode has to travel with
    // the counts or a user's own choice gets reported as a problem.
    assert.match(call, /gpu_memory_mode/);
  }
});

test("every auto-load path forwards the load response's fallback reason", () => {
  const calls = source.match(/\n\s*showAutoLoadSuccess\([\s\S]*?\);/g) ?? [];
  assert.ok(calls.length > 0, "no showAutoLoadSuccess call sites found");
  for (const call of calls) {
    assert.match(call, /cpu_fallback_reason/);
  }
});
