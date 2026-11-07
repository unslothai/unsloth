// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  readSrc,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();
const { GPU_LAYERS_AUTO, gpuLayersPatch, moeCpuOffloadApplies } = await import(
  "../src/features/chat/lib/gpu-placement.ts"
);
const { DEFAULT_PER_MODEL_CONFIG, normalizePerModelConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);

const manual = (gpuLayers: number, nCpuMoe?: number) => ({
  ...DEFAULT_PER_MODEL_CONFIG,
  gpuMemoryMode: "manual" as const,
  gpuLayers,
  nCpuMoe,
});

test("MoE CPU offload applies only with layers on the GPU", () => {
  assert.equal(moeCpuOffloadApplies(GPU_LAYERS_AUTO), false);
  assert.equal(moeCpuOffloadApplies(0), false);
  assert.equal(moeCpuOffloadApplies(1), true);
  assert.equal(moeCpuOffloadApplies(31), true);
});

test("setting GPU Layers to 0 drops the MoE CPU count", () => {
  const next = { ...manual(31, 10), ...gpuLayersPatch(0) };
  assert.equal(next.gpuLayers, 0);
  assert.equal(next.nCpuMoe, undefined);
});

test("other GPU Layers values keep the MoE CPU count", () => {
  assert.equal({ ...manual(31, 10), ...gpuLayersPatch(12) }.nCpuMoe, 10);
  // Auto keeps it hidden but stored, as before; only the positional split resets.
  assert.deepEqual(gpuLayersPatch(GPU_LAYERS_AUTO), {
    gpuLayers: GPU_LAYERS_AUTO,
    tensorSplit: null,
  });
  assert.equal({ ...manual(31, 10), ...gpuLayersPatch(GPU_LAYERS_AUTO) }.nCpuMoe, 10);
});

test("a saved 0-layer config loads without its stale MoE CPU count", () => {
  const loaded = normalizePerModelConfig(manual(0, 10));
  assert.equal(loaded.gpuLayers, 0);
  assert.equal(loaded.nCpuMoe, undefined);
  assert.equal(normalizePerModelConfig(manual(12, 10)).nCpuMoe, 10);
});

test("a 0-layer config is never sent or read back with n_cpu_moe", () => {
  const payload = toApiOverride(normalizePerModelConfig(manual(0, 10)));
  assert.equal(payload.gpu_layers, 0);
  assert.equal("n_cpu_moe" in payload, false);
  // A server override saved by an older client still loads without it.
  const restored = fromApiOverride(
    { gpu_memory_mode: "manual", gpu_layers: 0, n_cpu_moe: 10 },
    DEFAULT_PER_MODEL_CONFIG,
  );
  assert.equal(restored.gpuLayers, 0);
  assert.equal(restored.nCpuMoe, undefined);
});

test("the editor hides the slider at 0 GPU layers and routes edits through the patch", () => {
  const src = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(
    src,
    /const showMoeSlider =\s*isManual && moeCpuOffloadApplies\(gpuLayers\) && moeLayersMax > 0;/,
  );
  assert.match(src, /onChange=\{\(v\) => update\(gpuLayersPatch\(v\)\)\}/);
  // A same-click Load commits the typed layers after the MoE input, so 0 must still drop it.
  const commit = src.slice(src.indexOf("pendingPatch.nCpuMoe = committedMoeLayers;"));
  assert.match(
    commit.slice(0, 400),
    /committedGpuLayers != null && !moeCpuOffloadApplies\(committedGpuLayers\)[\s\S]*Object\.assign\(pendingPatch, gpuLayersPatch\(committedGpuLayers\)\)/,
  );
});
