// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { offloadCountsFrom, offloadWarning } = await import(
  "../src/features/chat/lib/partial-offload.ts"
);

test("a split load is reported", () => {
  assert.match(
    offloadWarning({ offloaded: 38, total: 60 })?.description ?? "",
    /^38 of 60 layers are on the GPU\./,
  );
  assert.equal(
    offloadWarning({ offloaded: 38, total: 60 })?.titleSuffix,
    ", partly on CPU",
  );
  assert.notEqual(offloadWarning({ offloaded: 1, total: 60 }), null);
});

test("a full offload is the normal case and says nothing", () => {
  assert.equal(offloadWarning({ offloaded: 60, total: 60 }), null);
});

test("no layers on the GPU is the worst case, not an excluded one", () => {
  const warning = offloadWarning({ offloaded: 0, total: 60 });
  assert.equal(warning?.titleSuffix, ", on CPU");
  assert.match(warning?.description ?? "", /^None of the 60 layers fit/);
});

test("a load that reported no counts says nothing", () => {
  assert.equal(offloadWarning({}), null);
  assert.equal(offloadWarning({ offloaded: null, total: null }), null);
  assert.equal(offloadWarning({ offloaded: 38, total: null }), null);
  assert.equal(offloadWarning({ offloaded: undefined, total: 60 }), null);
});

test("a nonsense total cannot produce a warning", () => {
  assert.equal(offloadWarning({ offloaded: 5, total: 0 }), null);
  assert.equal(offloadWarning({ offloaded: 5, total: -1 }), null);
  assert.equal(offloadWarning({ offloaded: 61, total: 60 }), null);
});

test("a split the user pinned themselves is not warned about", () => {
  assert.equal(
    offloadWarning({
      offloaded: 20,
      total: 60,
      gpuMemoryMode: "manual",
      gpuLayers: 20,
    }),
    null,
  );
  assert.notEqual(
    offloadWarning({ offloaded: 20, total: 60, gpuMemoryMode: "auto" }),
    null,
  );
  assert.notEqual(offloadWarning({ offloaded: 20, total: 60 }), null);
});

test("an -ngl passed through extras counts as the user's own choice", () => {
  assert.equal(
    offloadWarning({
      offloaded: 20,
      total: 60,
      gpuMemoryMode: "auto",
      offloadOverridden: true,
    }),
    null,
  );
  assert.equal(
    offloadWarning({ offloaded: 0, total: 60, offloadOverridden: true }),
    null,
  );
});

test("every load path reads the response the same way", () => {
  assert.deepEqual(
    offloadCountsFrom({
      // biome-ignore lint/style/useNamingConvention: api schema
      offloaded_layers: 38,
      // biome-ignore lint/style/useNamingConvention: api schema
      offload_total_layers: 60,
      // biome-ignore lint/style/useNamingConvention: api schema
      gpu_memory_mode: "auto",
      // biome-ignore lint/style/useNamingConvention: api schema
      gpu_layers: -1,
      // biome-ignore lint/style/useNamingConvention: api schema
      offload_overridden: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      cpu_fallback_reason: null,
      // biome-ignore lint/style/useNamingConvention: api schema
      gpu_backend_unavailable: false,
    }),
    {
      offloaded: 38,
      total: 60,
      gpuMemoryMode: "auto",
      gpuLayers: -1,
      offloadOverridden: false,
      cpuFallbackReason: null,
      gpuBackendUnavailable: false,
    },
  );
});

test("Manual mode with GPU Layers on Auto is still an automatic spill", () => {
  assert.notEqual(
    offloadWarning({
      offloaded: 20,
      total: 60,
      gpuMemoryMode: "manual",
      gpuLayers: -1,
    }),
    null,
  );
  assert.notEqual(
    offloadWarning({ offloaded: 0, total: 60, gpuMemoryMode: "manual" }),
    null,
  );
});

test("a GPU that llama.cpp could not use is not a size problem", () => {
  const broken = offloadWarning({
    offloaded: 0,
    total: 60,
    gpuBackendUnavailable: true,
  });
  assert.match(broken?.description ?? "", /could not use it/);
  assert.doesNotMatch(broken?.description ?? "", /may leave room|may let more/);
  assert.match(
    offloadWarning({ offloaded: 0, total: 60 })?.description ?? "",
    /None of the 60 layers fit/,
  );
  assert.match(
    offloadWarning({ offloaded: 20, total: 60, gpuBackendUnavailable: true })
      ?.description ?? "",
    /^20 of 60 layers/,
  );
});

test("a known reason for the CPU wins over the counts", () => {
  const warning = offloadWarning({
    offloaded: 0,
    total: 60,
    cpuFallbackReason: "vulkan_startup_crash",
  });
  assert.equal(warning?.titleSuffix, " on CPU");
  assert.match(warning?.description ?? "", /Vulkan backend crashed/);
  assert.doesNotMatch(warning?.description ?? "", /smaller quantization/);
  assert.equal(
    offloadWarning({ offloaded: 0, total: 60, cpuFallbackReason: "something" }),
    null,
  );
});

test("a requested GPU split that got no GPU at all still warns", () => {
  for (const counts of [
    { gpuMemoryMode: "manual", gpuLayers: 20 },
    { gpuMemoryMode: "auto", gpuLayers: -1, offloadOverridden: true },
  ]) {
    const warning = offloadWarning({
      offloaded: 0,
      total: 60,
      gpuBackendUnavailable: true,
      ...counts,
    });
    assert.match(warning?.description ?? "", /could not use it/);
  }
  assert.equal(
    offloadWarning({
      offloaded: 0,
      total: 60,
      gpuBackendUnavailable: true,
      gpuMemoryMode: "manual",
      gpuLayers: 0,
    }),
    null,
  );
});

test("the advice does not promise that a smaller quantization fits", () => {
  for (const offloaded of [0, 38]) {
    const text = offloadWarning({ offloaded, total: 60 })?.description ?? "";
    assert.match(text, /smaller quantization or a shorter context may/);
    assert.doesNotMatch(text, /would fit entirely|would leave room/);
  }
});
