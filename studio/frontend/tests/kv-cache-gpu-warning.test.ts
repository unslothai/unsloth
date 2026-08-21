import assert from "node:assert/strict";
import test from "node:test";

import { shouldWarnKvCacheGpuFallback } from "../src/features/model-picker/model-config/kv-cache-gpu-warning.ts";

const base = {
  cacheType: "q4_1",
  gpuLayers: -1,
  isDiffusion: false,
} as const;

test("KV cache advisory covers affected CUDA/HIP placement modes", () => {
  for (const backend of ["cuda", "rocm"]) {
    for (const gpuMemoryMode of ["auto", "manual"] as const) {
      for (const gpuLayers of gpuMemoryMode === "auto" ? [-1] : [-1, 1]) {
        assert.equal(
          shouldWarnKvCacheGpuFallback({
            ...base,
            backend,
            gpuMemoryMode,
            gpuLayers,
          }),
          true,
        );
      }
    }
  }
});

test("KV cache advisory preserves silent negative space", () => {
  for (const backend of ["vulkan", "metal", "cpu", "unknown", null]) {
    assert.equal(
      shouldWarnKvCacheGpuFallback({ ...base, backend, gpuMemoryMode: "auto" }),
      false,
    );
  }
  assert.equal(
    shouldWarnKvCacheGpuFallback({
      ...base,
      backend: "cuda",
      gpuMemoryMode: "manual",
      gpuLayers: 0,
    }),
    false,
  );
  assert.equal(
    shouldWarnKvCacheGpuFallback({
      ...base,
      backend: "cuda",
      cacheType: "q8_0",
      gpuMemoryMode: "auto",
    }),
    false,
  );
  assert.equal(
    shouldWarnKvCacheGpuFallback({
      ...base,
      backend: "cuda",
      isDiffusion: true,
      gpuMemoryMode: "auto",
    }),
    false,
  );
});
