// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

/** KV cache dtype, seed and GPU memory limit for vLLM / SGLang loads. */

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { normalizePerModelConfig, isDefaultConfig, DEFAULT_PER_MODEL_CONFIG } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const { residentRuntimeMatchesConfig } = await import(
  "../src/features/chat/lib/resident-config-match.ts"
);

test("stored launch options survive normalization and bad ones fall back to unset", () => {
  const kept = normalizePerModelConfig({
    engine: "vllm",
    engineKvCacheDtype: "fp8",
    engineSeed: 0,
    engineGpuMemoryUtilization: 0.5,
  });
  assert.equal(kept.engineKvCacheDtype, "fp8");
  assert.equal(kept.engineSeed, 0);
  assert.equal(kept.engineGpuMemoryUtilization, 0.5);
  assert.equal(isDefaultConfig(kept), false);

  const dropped = normalizePerModelConfig({
    engineKvCacheDtype: "fp4",
    engineSeed: -1,
    engineGpuMemoryUtilization: 1.5,
  });
  assert.equal(dropped.engineKvCacheDtype, "auto");
  assert.equal(dropped.engineSeed, null);
  assert.equal(dropped.engineGpuMemoryUtilization, null);
  assert.equal(isDefaultConfig(normalizePerModelConfig({})), true);
});

test("a changed launch option changes the loaded-config signature", () => {
  const base = loadedConfigSignature(DEFAULT_PER_MODEL_CONFIG);
  for (const patch of [
    { engineKvCacheDtype: "fp8" as const },
    { engineSeed: 1 },
    { engineGpuMemoryUtilization: 0.7 },
  ]) {
    assert.notEqual(loadedConfigSignature({ ...DEFAULT_PER_MODEL_CONFIG, ...patch }), base);
  }
});

test("a resident engine launched with other options does not satisfy the pick", () => {
  const standing = {
    speculativeType: "auto",
    gpuMemoryMode: "auto" as const,
    gpuLayers: -1,
    nCpuMoe: 0,
    reconcileGpuIds: (ids: number[] | null) => ids,
    resolveContextLength: (value: number | null) => value ?? 0,
    parallelSlots: 1,
    splitRatio: null,
    normalizeSpeculative: (value: string | null | undefined) => value ?? null,
  };
  const config = { ...DEFAULT_PER_MODEL_CONFIG, engine: "vllm" as const, engineSeed: 3 };
  const status = { engine: "vllm" as const, engine_seed: 3 };
  assert.equal(residentRuntimeMatchesConfig(status, config, standing), true);
  assert.equal(
    residentRuntimeMatchesConfig({ ...status, engine_seed: 4 }, config, standing),
    false,
  );
  assert.equal(
    residentRuntimeMatchesConfig(
      { ...status, engine_kv_cache_dtype: "fp8" },
      config,
      standing,
    ),
    false,
  );
});
