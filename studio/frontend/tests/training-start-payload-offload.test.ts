// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Offload layers in the start payload: off by default, a full finetune always sends it off,
// and the VRAM budget goes out only with "auto", the one mode that sizes to it.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { offloadHardwareSupported, offloadPayload, offloadSupported } = await import("../src/features/training/api/mappers.ts");
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

test("offload goes out off wherever the run cannot offload", () => {
  const on = { ...base, offloadLayers: "auto" as const, offloadVramGb: 8 };
  for (const blocked of [
    { gradientCheckpointing: "none" as const },
    { isEmbeddingModel: true },
    { isAudioModel: true },
    { modelType: "decision" as const },
    { modelType: "embeddings" as const },
  ]) {
    assert.equal(offloadSupported({ ...on, ...blocked }), false);
    assert.deepEqual(offloadPayload({ ...on, ...blocked }).offload_layers, 0);
    assert.equal(offloadPayload({ ...on, ...blocked }).offload_vram_gb, null);
  }
  assert.equal(offloadSupported(on), true);
});

test("a count above the backend limit is clamped instead of failing the start", () => {
  assert.equal(offloadPayload({ ...base, offloadLayers: 5000 }).offload_layers, 1024);
});

test("offload needs a CUDA or ROCm card that is not a unified-memory APU", () => {
  const sys = (device_backend: string, unified: boolean[]) => ({
    status: "ready" as const,
    device_backend: device_backend as "cuda",
    gpu: { available: true, devices: unified.map((u, index) => ({ index, unified_memory: u })) },
  });
  assert.equal(offloadHardwareSupported(sys("cuda", [false])), true);
  assert.equal(offloadHardwareSupported(sys("rocm", [false])), true);
  for (const backend of ["xpu", "mlx", "cpu"]) {
    assert.equal(offloadHardwareSupported(sys(backend, [false])), false);
  }
  // A ROCm APU alone (Strix Halo) has no separate pool; beside a discrete card it does.
  assert.equal(offloadHardwareSupported(sys("rocm", [true])), false);
  assert.equal(offloadHardwareSupported(sys("rocm", [true, false])), true);
  assert.equal(offloadHardwareSupported(null), true);
  assert.equal(offloadHardwareSupported({ ...sys("cpu", []), status: "pending" }), true);
  // A hidden setting is sent off too, so a saved Count cannot fail the run.
  const saved = { ...base, offloadLayers: 14 };
  assert.equal(offloadPayload(saved, sys("xpu", [false])).offload_layers, 0);
  assert.equal(offloadPayload(saved, sys("rocm", [true])).offload_layers, 0);
  assert.equal(offloadPayload(saved, sys("cuda", [false])).offload_layers, 14);
});
