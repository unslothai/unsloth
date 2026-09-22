// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

// Execute the shipped status patch and rollback payload, like reasoning-budget-rollback.test.ts.
const applier = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
const start = applier.indexOf("    residentCheckpoint: checkpointId,");
const end = applier.indexOf("  });", start);
assert.ok(start >= 0 && end > start);
const hydrate = new Function(
  "checkpointId",
  "status",
  `return { ${applier.slice(start, end)} };`,
) as (checkpointId: string, status: Record<string, unknown>) => Record<string, unknown>;

const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const rollbackStart = runtime.indexOf("engine: stateBeforeUnload.loadedEngine,");
const rollbackEnd = runtime.indexOf("is_lora: previousIsLora,", rollbackStart);
assert.ok(rollbackStart >= 0 && rollbackEnd > rollbackStart);
const rollback = new Function(
  "stateBeforeUnload",
  "rollbackNativePathLease",
  "hfToken",
  "rollbackMaxSeqLength",
  `return { ${runtime.slice(rollbackStart, rollbackEnd)} };`,
) as (state: Record<string, unknown>, ...rest: unknown[]) => Record<string, unknown>;

test("a failed switch restores the managed resident's precision and parallel mode", () => {
  const resident = hydrate("org/model", {
    engine: "sglang",
    engine_precision: "bf16",
    engine_parallelism: "pipeline",
  });
  const request = rollback(resident, undefined, null, 4096);
  assert.equal(request.engine, "sglang");
  assert.equal(request.engine_precision, "bf16");
  assert.equal(request.engine_parallelism, "pipeline");
  // The backend reads an explicit 4-bit request without a precision as INT4.
  assert.equal(request.load_in_4bit, false);
});

test("a default-backend resident still rolls back with 4-bit loading", () => {
  const request = rollback(hydrate("org/model", {}), undefined, null, 4096);
  assert.equal(request.engine, "auto");
  assert.equal(request.engine_precision, "auto");
  assert.equal(request.engine_parallelism, "tensor");
  assert.equal(request.load_in_4bit, true);
});
