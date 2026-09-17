// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveResidentEstimateRequest,
  selectResidentEstimateSettings,
} from "../src/features/model-picker/model-config/resident-memory-request.ts";

const loaded: Parameters<typeof selectResidentEstimateSettings>[0] = {
  modelLoading: false,
  loadedKvCacheDtype: "q8_0",
  loadedNParallel: 2,
  loadedNBatch: 512,
  loadedNUbatch: 128,
  loadedCtxCheckpoints: 8,
  loadedSpeculativeType: "off",
  loadedSpecDraftNMax: 3,
  loadedSpecDraftCacheDtype: "q8_0",
  loadedTensorParallel: false,
  loadedDisableVision: true,
  loadedGpuMemoryMode: "auto",
  loadedGpuLayers: null,
  loadedNCpuMoe: null,
  loadedGpuIds: [0],
  loadedLlamaExtraArgs: ["--no-kv-offload"],
  loadedCpuFallback: false,
  specFallbackReason: null,
  mmprojFallbackReason: null,
};

test("pending controls cannot change the loaded placement or memory requirements", () => {
  const pending = {
    modelPath: "model",
    ggufVariant: "Q4_K_M",
    nCtx: 262144,
    cacheTypeKv: "f16",
    nParallel: 8,
    nBatch: 4096,
    nUbatch: 2048,
    ctxCheckpoints: 64,
    speculativeType: "mtp",
    specDraftNMax: 16,
    specDraftCacheType: "f16",
    tensorParallel: true,
    disableVision: false,
    gpuMemoryMode: "manual",
    gpuLayers: 99,
    nCpuMoe: 0,
    selectedGpuIds: [1],
    llamaExtraArgs: ["--swa-full"],
  };
  const request = resolveResidentEstimateRequest(
    pending,
    selectResidentEstimateSettings({ ...pending, ...loaded }),
    8192,
  );
  assert.deepEqual(request, {
    modelPath: "model",
    ggufVariant: "Q4_K_M",
    hfToken: undefined,
    nativePathToken: undefined,
    nCtx: 8192,
    cacheTypeKv: "q8_0",
    nParallel: 2,
    nBatch: 512,
    nUbatch: 128,
    ctxCheckpoints: 8,
    speculativeType: "off",
    specDraftNMax: 3,
    specDraftCacheType: "q8_0",
    tensorParallel: false,
    disableVision: true,
    gpuMemoryMode: "auto",
    gpuLayers: null,
    nCpuMoe: null,
    selectedGpuIds: [0],
    llamaExtraArgs: ["--no-kv-offload"],
  });
  const defaults = resolveResidentEstimateRequest(
    pending,
    selectResidentEstimateSettings({
      ...loaded,
      loadedNParallel: null,
      loadedLlamaExtraArgs: null,
    }),
    8192,
  );
  assert.equal(defaults?.nParallel, null);
  assert.equal(defaults?.llamaExtraArgs, null);
});

test("missing baselines and recorded fallbacks withhold resident credit", () => {
  for (const patch of [
    { modelLoading: true },
    { loadedGpuMemoryMode: null },
    { loadedSpeculativeType: null },
    { loadedTensorParallel: null },
    { loadedDisableVision: null },
    { loadedGpuMemoryMode: "manual" as const, loadedGpuLayers: null, loadedNCpuMoe: 0 },
    { loadedGpuMemoryMode: "manual" as const, loadedGpuLayers: 12, loadedNCpuMoe: null },
    { loadedCpuFallback: true },
    { specFallbackReason: "runtime_error" },
    { mmprojFallbackReason: "cpu_offload" as const },
    { mmprojFallbackReason: "projector_startup_failure" as const },
  ]) {
    assert.equal(selectResidentEstimateSettings({ ...loaded, ...patch }), null);
  }
  const settings = selectResidentEstimateSettings(loaded);
  for (const context of [null, 0, Number.NaN]) {
    assert.equal(
      resolveResidentEstimateRequest(
        { modelPath: "model", nCtx: 262144 },
        settings,
        context,
      ),
      null,
    );
  }
  assert.equal(resolveResidentEstimateRequest(null, settings, 8192), null);
});

test("fixed Manual layers and automatic layers retain their loaded meaning", () => {
  for (const layers of [12, -1]) {
    const settings = selectResidentEstimateSettings({
      ...loaded,
      loadedGpuMemoryMode: "manual",
      loadedGpuLayers: layers,
      loadedNCpuMoe: 4,
    });
    assert.equal(settings?.gpuMemoryMode, "manual");
    assert.equal(settings?.gpuLayers, layers < 0 ? null : layers);
    assert.equal(settings?.nCpuMoe, 4);
  }
});
