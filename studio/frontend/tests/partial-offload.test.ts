// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { expertSpillNote, offloadCountsFrom, offloadWarning } = await import(
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
      // biome-ignore lint/style/useNamingConvention: api schema
      overflow_layers: 7,
      // biome-ignore lint/style/useNamingConvention: api schema
      expert_cache_mib: null,
      // biome-ignore lint/style/useNamingConvention: api schema
      gpu_weights_mib: 1228.8,
      // biome-ignore lint/style/useNamingConvention: api schema
      host_weights_mib: 2048,
    }),
    {
      offloaded: 38,
      total: 60,
      gpuMemoryMode: "auto",
      gpuLayers: -1,
      offloadOverridden: false,
      cpuFallbackReason: null,
      gpuBackendUnavailable: false,
      overflowLayers: 7,
      expertCacheMib: null,
      gpuWeightsMib: 1228.8,
      hostWeightsMib: 2048,
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
  for (const [reason, text] of [
    ["vulkan_no_device", /No usable Vulkan GPU was found/],
    ["vulkan_startup_stall", /stopped responding during startup/],
  ] as const) {
    const other = offloadWarning({ offloaded: 0, total: 60, cpuFallbackReason: reason });
    assert.equal(other?.titleSuffix, " on CPU");
    assert.match(other?.description ?? "", text);
  }
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

test("the notice says how much of the weights is on the GPU and on the CPU", () => {
  // A T4 load of a 27B: CUDA0 12677.16 MiB, CPU_Mapped 5833.19 MiB.
  assert.match(
    offloadWarning({ offloaded: 45, total: 66, gpuWeightsMib: 12677.16, hostWeightsMib: 5833.19 })
      ?.description ?? "",
    /may let more of it fit on the GPU\. Model weights: 12 GiB on the GPU, 5\.7 GiB on the CPU\.$/,
  );
  assert.match(
    offloadWarning({ offloaded: 0, total: 60, gpuWeightsMib: 0, hostWeightsMib: 22400 })
      ?.description ?? "",
    /may leave room on the GPU\. Model weights: 22 GiB on the CPU\.$/,
  );
  assert.match(
    offloadWarning({ offloaded: 9, total: 25, gpuWeightsMib: 40 })?.description ?? "",
    /Model weights: 40 MiB on the GPU\.$/,
  );
  assert.doesNotMatch(
    offloadWarning({ offloaded: 54, total: 66, gpuWeightsMib: 19456, hostWeightsMib: 4813 })
      ?.description ?? "",
    /VRAM|RAM/,
  );
});

test("missing or nonsense weight figures leave the notice as it was", () => {
  for (const weights of [
    {},
    { gpuWeightsMib: null, hostWeightsMib: null },
    { gpuWeightsMib: 0, hostWeightsMib: 0 },
    { gpuWeightsMib: -5, hostWeightsMib: Number.NaN },
  ]) {
    assert.match(
      offloadWarning({ offloaded: 38, total: 60, ...weights })?.description ?? "",
      /may let more of it fit on the GPU\.$/,
    );
  }
});

test("expert weights the fit kept on the CPU are stated, not warned about", () => {
  // gemma-4-26B-A4B Q8_0 on a 3090: llama.cpp logs "offloaded 31/31" with seven layers' experts in CUDA_Host.
  const spill = {
    offloaded: 31,
    total: 31,
    gpuMemoryMode: "auto",
    gpuLayers: -1,
    overflowLayers: 7,
    gpuWeightsMib: 20709.15,
    hostWeightsMib: 5639.4,
  };
  assert.equal(offloadWarning(spill), null);
  assert.equal(
    expertSpillNote(spill),
    "All 31 layers are on the GPU. The expert weights of 7 layers are on the CPU. Model weights: 20 GiB on the GPU, 5.5 GiB on the CPU.",
  );
  assert.match(expertSpillNote({ ...spill, overflowLayers: 1 }) ?? "", /of 1 layer are on the CPU/);
  assert.equal(
    expertSpillNote({ offloaded: 31, total: 31, overflowLayers: 30, expertCacheMib: 6144 }),
    "All 31 layers are on the GPU. Every expert is on the CPU, with a 6.0 GiB cache of the most used ones on the GPU.",
  );
});

test("only a spill nobody chose, on an otherwise full load, gets the expert note", () => {
  const spill = { offloaded: 31, total: 31, overflowLayers: 7 };
  assert.equal(expertSpillNote({ offloaded: 31, total: 31, overflowLayers: 0 }), null);
  assert.equal(expertSpillNote({ offloaded: 31, total: 31 }), null);
  assert.equal(expertSpillNote({ ...spill, gpuMemoryMode: "manual", gpuLayers: 31 }), null);
  assert.equal(expertSpillNote({ ...spill, offloadOverridden: true }), null);
  // Manual with GPU Layers on Auto is still the fit's doing.
  assert.notEqual(expertSpillNote({ ...spill, gpuMemoryMode: "manual", gpuLayers: -1 }), null);
  assert.equal(expertSpillNote({ ...spill, offloaded: 20 }), null);
  assert.equal(expertSpillNote({ ...spill, cpuFallbackReason: "vulkan_startup_crash" }), null);
  assert.equal(expertSpillNote({}), null);
});

test("a full count with every buffer in system memory is not a success", () => {
  const hostOnly = { offloaded: 29, total: 29, gpuWeightsMib: 0, hostWeightsMib: 499.8 };
  const broken = offloadWarning({ ...hostOnly, gpuBackendUnavailable: true });
  assert.equal(broken?.titleSuffix, ", on CPU");
  assert.match(broken?.description ?? "", /could not use it/);
  // The same buffers on a machine with no GPU to miss are the expected placement.
  assert.equal(offloadWarning(hostOnly), null);
});
