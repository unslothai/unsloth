// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import { engineLaunchFieldsFrom } from "../src/features/chat/lib/engine-launch-fields.ts";

const applier = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
const start = applier.indexOf("    residentCheckpoint: checkpointId,");
const end = applier.indexOf("  });", start);
assert.ok(start >= 0 && end > start);
const hydrateBody = new Function(
  "checkpointId",
  "status",
  "engineLaunchFieldsFrom",
  `return { ${applier.slice(start, end)} };`,
);
const hydrate = (checkpointId: string, status: Record<string, unknown>) =>
  hydrateBody(checkpointId, status, engineLaunchFieldsFrom) as Record<string, unknown>;

const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const rollbackStart = runtime.indexOf("engine: stateBeforeUnload.loadedEngine,");
const rollbackEnd = runtime.indexOf("is_lora: previousIsLora,", rollbackStart);
assert.ok(rollbackStart >= 0 && rollbackEnd > rollbackStart);
const rollback = new Function(
  "stateBeforeUnload",
  "rollbackNativePathLease",
  "hfToken",
  "rollbackMaxSeqLength",
  "rollbackConfig",
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
  assert.equal(request.engine_kv_cache_dtype, "auto");
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

test("a default-backend resident loaded in 16-bit rolls back in 16-bit", () => {
  const resident = hydrate("org/model", { load_in_4bit: false });
  const request = rollback(resident, undefined, null, 4096, { loadIn4bit: false });
  assert.equal(request.engine, "auto");
  assert.equal(request.load_in_4bit, false);
});

test("a failed switch restores the managed resident's KV cache, seed and memory limit", () => {
  const resident = hydrate("org/model", {
    engine: "vllm",
    engine_kv_cache_dtype: "fp8",
    engine_seed: 0,
    engine_gpu_memory_utilization: 0.5,
  });
  const request = rollback(resident, undefined, null, 4096);
  assert.equal(request.engine_kv_cache_dtype, "fp8");
  assert.equal(request.engine_seed, 0);
  assert.equal(request.engine_gpu_memory_utilization, 0.5);
});
