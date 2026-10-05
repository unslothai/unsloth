// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Offload layers in the start payload: off by default, a full finetune always sends it off,
// and the VRAM budget goes out only with "auto", the one mode that sizes to it.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { offloadPayload } = await import("../src/features/training/api/mappers.ts");
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);

const base = { ...initialTrainingConfigState, trainingMethod: "lora" as const };

test("defaults send offload off at depth 2", () => {
  assert.deepEqual(offloadPayload(base), {
    offload_layers: 0,
    offload_vram_gb: null,
    prefetch_depth: 2,
  });
});

test("auto carries the budget and an auto depth", () => {
  assert.deepEqual(
    offloadPayload({ ...base, offloadLayers: "auto", offloadVramGb: 11.5, prefetchDepth: "auto" }),
    { offload_layers: "auto", offload_vram_gb: 11.5, prefetch_depth: "auto" },
  );
});

test("a fixed count drops the budget and rounds the count", () => {
  assert.deepEqual(offloadPayload({ ...base, offloadLayers: 14.7, offloadVramGb: 8, prefetchDepth: 3 }), {
    offload_layers: 14,
    offload_vram_gb: null,
    prefetch_depth: 3,
  });
});

test("a full finetune never offloads", () => {
  const out = offloadPayload({ ...base, trainingMethod: "full", offloadLayers: "auto", offloadVramGb: 8 });
  assert.equal(out.offload_layers, 0);
  assert.equal(out.offload_vram_gb, null);
});

test("depth is clamped to what the backend accepts", () => {
  assert.equal(offloadPayload({ ...base, offloadLayers: 4, prefetchDepth: 40 }).prefetch_depth, 8);
  assert.equal(offloadPayload({ ...base, offloadLayers: 4, prefetchDepth: 0 }).prefetch_depth, 2);
});
