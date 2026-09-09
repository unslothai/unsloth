// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveResidentEstimateRequest,
  selectResidentEstimateSettings,
} from "../src/features/model-picker/model-config/resident-memory-request.ts";

const LOADED: Parameters<typeof selectResidentEstimateSettings>[0] = {
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

test("staged Manual controls cannot turn an Auto resident into fixed placement", () => {
  const staged = {
    ...LOADED,
    gpuMemoryMode: "manual",
    gpuLayers: 99,
    nCpuMoe: 0,
  };
  const request = resolveResidentEstimateRequest(
    { modelPath: "model", gpuMemoryMode: "manual", gpuLayers: 99 },
    selectResidentEstimateSettings(staged),
    8192,
  );
  assert.ok(request);
  assert.equal(request.gpuMemoryMode, "auto");
  assert.equal(request.gpuLayers, null);
});

test("resident sizing reads loaded baselines despite edits to every control", () => {
  const staged = {
    ...LOADED,
    loadedGpuMemoryMode: "manual" as const,
    loadedGpuLayers: 12,
    loadedNCpuMoe: 4,
    kvCacheDtype: "f16",
    nParallel: 8,
    nBatch: 4096,
    nUbatch: 2048,
    ctxCheckpoints: 64,
    speculativeType: "mtp",
    specDraftNMax: 16,
    specDraftCacheDtype: "f16",
    tensorParallel: true,
    disableVision: false,
    gpuMemoryMode: "auto",
    gpuLayers: -1,
    nCpuMoe: 0,
    selectedGpuIds: [1],
  };
  const request = resolveResidentEstimateRequest(
    {
      modelPath: "model",
      ggufVariant: "Q4_K_M",
      hfToken: "test-token",
      nativePathToken: "test-path",
      nCtx: 262144,
      ...staged,
    },
    selectResidentEstimateSettings(staged),
    8192,
  );
  assert.deepEqual(request, {
    modelPath: "model",
    ggufVariant: "Q4_K_M",
    hfToken: "test-token",
    nativePathToken: "test-path",
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
    gpuMemoryMode: "manual",
    gpuLayers: 12,
    nCpuMoe: 4,
    selectedGpuIds: [0],
    llamaExtraArgs: ["--no-kv-offload"],
  });
});

test("loaded defaults never fall back to pending values", () => {
  const state = {
    ...LOADED,
    loadedKvCacheDtype: null,
    loadedNParallel: null,
    loadedNBatch: null,
    loadedNUbatch: null,
    loadedCtxCheckpoints: null,
    loadedSpecDraftNMax: null,
    loadedSpecDraftCacheDtype: null,
    loadedGpuIds: null,
    loadedLlamaExtraArgs: null,
  };
  const request = resolveResidentEstimateRequest(
    {
      modelPath: "model",
      cacheTypeKv: "q4_0",
      nParallel: 8,
      nBatch: 4096,
      nUbatch: 2048,
      ctxCheckpoints: 64,
      specDraftNMax: 16,
      specDraftCacheType: "q4_0",
      selectedGpuIds: [1],
      llamaExtraArgs: ["--swa-full"],
    },
    selectResidentEstimateSettings(state),
    8192,
  );
  for (const field of [
    "cacheTypeKv",
    "nParallel",
    "nBatch",
    "nUbatch",
    "ctxCheckpoints",
    "specDraftNMax",
    "specDraftCacheType",
    "selectedGpuIds",
    "llamaExtraArgs",
  ] as const) {
    assert.equal(request?.[field], null, field);
  }
});

test("unhydrated resident baselines withhold the estimate", () => {
  for (const field of [
    "loadedGpuMemoryMode",
    "loadedSpeculativeType",
    "loadedTensorParallel",
    "loadedDisableVision",
  ] as const) {
    assert.equal(
      selectResidentEstimateSettings({ ...LOADED, [field]: null }),
      null,
      field,
    );
  }
  for (const field of ["loadedGpuLayers", "loadedNCpuMoe"] as const) {
    assert.equal(
      selectResidentEstimateSettings({
        ...LOADED,
        loadedGpuMemoryMode: "manual",
        loadedGpuLayers: 12,
        loadedNCpuMoe: 0,
        [field]: null,
      }),
      null,
      field,
    );
  }
  assert.equal(
    selectResidentEstimateSettings({
      ...LOADED,
      loadedGpuMemoryMode: "manual",
      loadedGpuLayers: -1,
      loadedNCpuMoe: 0,
    })?.gpuLayers,
    null,
  );
});

test("resident fallbacks still withhold estimates built from requested baselines", () => {
  for (const specFallbackReason of [
    "runtime_error",
    "binary_outdated",
    "mtp_partial_offload",
  ]) {
    assert.equal(
      selectResidentEstimateSettings({ ...LOADED, specFallbackReason }),
      null,
    );
  }
  for (const mmprojFallbackReason of [
    "cpu_offload",
    "projector_incompatible",
    "projector_startup_failure",
  ] as const) {
    assert.equal(
      selectResidentEstimateSettings({ ...LOADED, mmprojFallbackReason }),
      null,
    );
  }
  assert.equal(
    selectResidentEstimateSettings({ ...LOADED, loadedCpuFallback: true }),
    null,
  );
});

test("resident pricing requires an active source and reported context", () => {
  const settings = selectResidentEstimateSettings(LOADED);
  assert.equal(resolveResidentEstimateRequest(null, settings, 8192), null);
  assert.equal(
    resolveResidentEstimateRequest({ modelPath: "model" }, null, 8192),
    null,
  );
  for (const context of [null, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(
      resolveResidentEstimateRequest(
        { modelPath: "model", nCtx: 262144 },
        settings,
        context,
      ),
      null,
    );
  }
});
