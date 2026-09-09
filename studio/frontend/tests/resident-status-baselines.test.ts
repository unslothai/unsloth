// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { selectResidentEstimateSettings } from "../src/features/model-picker/model-config/resident-memory-request.ts";

// Execute the shipped status patch without importing the browser store graph.
const source = readFileSync(
  new URL(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
    import.meta.url,
  ),
  "utf8",
);
const start = source.indexOf("    // Controls follow the server while clean;");
const end = source.indexOf(
  "    // A load knob like tensorParallel above.",
  start,
);
assert.ok(start >= 0 && end > start);
const apply = new Function(
  "prevState",
  "status",
  "seedLoadParams",
  "hydratingExistingModel",
  "currentSpecType",
  `return { ...prevState, ${source.slice(start, end)} };`,
) as (
  previous: typeof loaded,
  status: Record<string, unknown>,
  settled: boolean,
  changed: boolean,
  specType: string,
) => typeof loaded;

const loaded = {
  modelLoading: false,
  kvCacheDtype: "f16",
  loadedKvCacheDtype: "f16",
  speculativeType: "off",
  loadedSpeculativeType: "off",
  specDraftNMax: 3,
  loadedSpecDraftNMax: 3,
  tensorParallel: false,
  loadedTensorParallel: false,
  loadedNParallel: 2,
  loadedNBatch: 512,
  loadedNUbatch: 128,
  loadedCtxCheckpoints: 8,
  loadedSpecDraftCacheDtype: "q8_0",
  loadedDisableVision: true,
  loadedGpuMemoryMode: "manual" as const,
  loadedGpuLayers: 12,
  loadedNCpuMoe: 0,
  loadedGpuIds: [0],
  loadedLlamaExtraArgs: [],
  loadedCpuFallback: false,
  specFallbackReason: null,
  mmprojFallbackReason: null,
};
const status = {
  cache_type_kv: "q4_0",
  speculative_type: "mtp",
  spec_draft_n_max: 8,
  tensor_parallel: true,
};

function checkBaselines(state: typeof loaded) {
  const settings = selectResidentEstimateSettings(state);
  assert.equal(settings?.cacheTypeKv, "q4_0");
  assert.equal(settings?.speculativeType, "mtp");
  assert.equal(settings?.specDraftNMax, 8);
  assert.equal(settings?.tensorParallel, true);
}

test("same-model external reload advances resident pricing and clean controls", () => {
  const next = apply(loaded, status, true, false, "mtp");
  checkBaselines(next);
  assert.equal(next.kvCacheDtype, "q4_0");
  assert.equal(next.speculativeType, "mtp");
  assert.equal(next.specDraftNMax, 8);
  assert.equal(next.tensorParallel, true);
});

test("pending edits survive external reloads and subsequent steady polls", () => {
  const dirty = {
    ...loaded,
    kvCacheDtype: "q8_0",
    speculativeType: "auto",
    specDraftNMax: 16,
    tensorParallel: true,
  };
  const next = apply(
    dirty,
    { ...status, tensor_parallel: false },
    true,
    false,
    "mtp",
  );
  assert.equal(next.loadedTensorParallel, false);
  assert.equal(next.tensorParallel, true);
  assert.equal(next.loadedKvCacheDtype, "q4_0");
  for (const key of [
    "kvCacheDtype",
    "speculativeType",
    "specDraftNMax",
    "tensorParallel",
  ] as const) {
    assert.equal(next[key], dirty[key]);
  }
  assert.deepEqual(
    apply(next, { ...status, tensor_parallel: false }, true, false, "mtp"),
    next,
  );
});

test("in-flight loads and absent echoes cannot overwrite recorded baselines", () => {
  assert.deepEqual(apply(loaded, status, false, false, "mtp"), loaded);
  assert.deepEqual(apply(loaded, {}, true, false, "auto"), loaded);
});

test("model or variant changes replace controls belonging to the old model", () => {
  const next = apply(
    {
      ...loaded,
      kvCacheDtype: "q8_0",
      speculativeType: "auto",
      specDraftNMax: 16,
    },
    status,
    true,
    true,
    "mtp",
  );
  checkBaselines(next);
  assert.equal(next.kvCacheDtype, "q4_0");
  assert.equal(next.speculativeType, "mtp");
  assert.equal(next.specDraftNMax, 8);
});

test("explicit null clears nullable baselines rather than keeping stale allocations", () => {
  const next = apply(
    loaded,
    { cache_type_kv: null, spec_draft_n_max: null },
    true,
    false,
    "off",
  );
  assert.equal(next.loadedKvCacheDtype, null);
  assert.equal(next.loadedSpecDraftNMax, null);
  assert.equal(next.kvCacheDtype, null);
  assert.equal(next.specDraftNMax, null);
});

test("speculative defaults still seed on adoption when older backends omit the mode", () => {
  const next = apply(loaded, {}, true, true, "auto");
  assert.equal(next.speculativeType, "auto");
  assert.equal(next.loadedSpeculativeType, "auto");
});
