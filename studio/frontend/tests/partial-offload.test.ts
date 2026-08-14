// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  offloadCountsFrom,
  offloadWarning,
} from "../src/features/chat/lib/partial-offload.ts";

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
  // cpu_fallback_reason is only ever set for the Vulkan startup-crash recovery,
  // so an ordinary --fit on load that got nothing onto the GPU has no other
  // reporting at all: it announced plain success and simply ran slowly.
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
  // Manual mode with an explicit layer count means the user chose that count.
  // The model may well fit, so "pick a smaller quantization" would be advice
  // against their own decision.
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
  // Absent (an older backend) is treated as automatic, which is the default.
  assert.notEqual(offloadWarning({ offloaded: 20, total: 60 }), null);
});

test("an -ngl passed through extras counts as the user's own choice", () => {
  // Auto mode respects an inherited -ngl rather than stripping it, so the mode
  // still reads "auto" for a split the user picked deliberately.
  assert.equal(
    offloadWarning({
      offloaded: 20,
      total: 60,
      gpuMemoryMode: "auto",
      offloadOverridden: true,
    }),
    null,
  );
  // Including the all-on-CPU case, which is a legitimate thing to ask for.
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
  // Manual + Auto layers hands placement back to llama.cpp exactly like Auto
  // mode does, and the response still reads "manual", so the mode alone is not
  // the question: the requested count is.
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
  // A failed CUDA/ROCm init logs the same 0/M line as a fit that placed no
  // layers, and telling a user with a broken install to pick a smaller
  // quantization sends them to re-download a model that was never the problem.
  const broken = offloadWarning({
    offloaded: 0,
    total: 60,
    gpuBackendUnavailable: true,
  });
  assert.match(broken?.description ?? "", /could not use it/);
  assert.doesNotMatch(broken?.description ?? "", /smaller quantization would/);
  // A genuine zero-layer fit still gets the size wording.
  assert.match(
    offloadWarning({ offloaded: 0, total: 60 })?.description ?? "",
    /None of the 60 layers fit/,
  );
  // It only speaks to the all-CPU case; a split load is a split load.
  assert.match(
    offloadWarning({ offloaded: 20, total: 60, gpuBackendUnavailable: true })
      ?.description ?? "",
    /^20 of 60 layers/,
  );
});

test("a known reason for the CPU wins over the counts", () => {
  // A recovered Vulkan startup crash leaves a 0/M line behind it. Reading that
  // as "nothing fit" would blame the model's size for a backend crash and
  // recommend a quantization that changes nothing.
  const warning = offloadWarning({
    offloaded: 0,
    total: 60,
    cpuFallbackReason: "vulkan_startup_crash",
  });
  assert.equal(warning?.titleSuffix, " on CPU");
  assert.match(warning?.description ?? "", /Vulkan backend crashed/);
  assert.doesNotMatch(warning?.description ?? "", /smaller quantization/);
  // An unrecognised reason is still a reason: say nothing rather than guess.
  assert.equal(
    offloadWarning({ offloaded: 0, total: 60, cpuFallbackReason: "something" }),
    null,
  );
});
