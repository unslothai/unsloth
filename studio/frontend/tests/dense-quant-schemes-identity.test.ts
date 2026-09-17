// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The media picker memoises its option list on `denseQuantSchemes` (useImageModels,
// curatedRowLabelFor), and the GGUF rows carry a host-dependent "(Slow)" suffix. So a snapshot
// differing only in free host RAM (re-probed by `refresh_memory=true`) must NOT hand back a fresh
// array, or every probe rebuilds the list and detaches every row it had just drawn.

import assert from "node:assert/strict";
import test from "node:test";
import type * as GpuHooks from "../src/hooks/use-gpu-info.ts";
import { normalizeDenseQuantSchemes } from "../src/lib/dense-quant-schemes.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const hooks = loadWithStubs<typeof GpuHooks>(
  new URL("../src/hooks/use-gpu-info.ts", import.meta.url),
  {
    react: {
      useState: (initial: unknown) => [
        typeof initial === "function" ? initial() : initial,
        () => {},
      ],
      useEffect: () => {},
      useMemo: (read: () => unknown) => read(),
    },
    "./use-system": { getCachedSystemInfo: () => null },
    "@/lib/dense-quant-schemes": { normalizeDenseQuantSchemes },
  },
  { relativePassthrough: true },
);

/** A GpuInfo carrying the given schemes and free-RAM reading; every other field is fixed. */
function snapshot(
  schemes: readonly string[],
  systemRamAvailableGb: number,
): GpuHooks.GpuInfo {
  return {
    available: true,
    budgetKnown: true,
    sharedMemory: false,
    unifiedMemory: false,
    backend: "cuda",
    denseQuantSupported: schemes.length > 0,
    denseQuantSchemes: schemes,
    name: "NVIDIA GeForce RTX 4090",
    memoryTotalGb: 24,
    memorySharedGb: 0,
    dedicatedMemoryTotalGb: 24,
    maxDeviceMemoryGb: 24,
    loadDeviceMemoryGb: 24,
    loadDeviceSharedMemory: false,
    loadDeviceSharesHostMemory: false,
    deviceCount: 1,
    cpuCore: 8,
    cpuThread: 16,
    systemRamAvailableGb,
    systemRamAvailableHostGb: systemRamAvailableGb,
    systemRamAvailableKnown: true,
    systemRamTotalGb: 64,
  };
}

test("the premise: each probe normalises a fresh schemes array", () => {
  const reported = ["fp8", "int8"];
  const first = normalizeDenseQuantSchemes(reported);
  const second = normalizeDenseQuantSchemes(reported);
  assert.deepEqual(first, second);
  assert.notEqual(first, second, "a fresh array per probe is what defeats the memo");
});

test("a probe that only moved free RAM keeps the schemes array identity", () => {
  const current = snapshot(normalizeDenseQuantSchemes(["fp8", "int8"]), 31.2);
  const next = snapshot(normalizeDenseQuantSchemes(["fp8", "int8"]), 28.7);
  assert.notEqual(
    current.denseQuantSchemes,
    next.denseQuantSchemes,
    "the probe really did build a new array",
  );

  const held = hooks.withStableSchemes(current, next);

  assert.equal(
    held.denseQuantSchemes,
    current.denseQuantSchemes,
    "the option list memo must not be invalidated by a RAM-only probe",
  );
});

test("the RAM reading the probe refreshed is still published", () => {
  const current = snapshot(normalizeDenseQuantSchemes(["fp8"]), 31.2);
  const next = snapshot(normalizeDenseQuantSchemes(["fp8"]), 28.7);

  const held = hooks.withStableSchemes(current, next);

  // Holding the array must not hold the snapshot: the footprint panel reads these.
  assert.equal(held.systemRamAvailableGb, 28.7);
  assert.equal(held.systemRamAvailableHostGb, 28.7);
});

test("a host that really changed its schemes gets the new array", () => {
  const current = snapshot(normalizeDenseQuantSchemes(["fp8", "int8"]), 31.2);
  const next = snapshot(normalizeDenseQuantSchemes(["int8"]), 31.2);

  const held = hooks.withStableSchemes(current, next);

  assert.equal(held.denseQuantSchemes, next.denseQuantSchemes);
  assert.deepEqual(held.denseQuantSchemes, ["int8"]);
});

test("order is part of the identity: schemes are ranked best first", () => {
  const current = snapshot(normalizeDenseQuantSchemes(["fp8", "int8"]), 31.2);
  const next = snapshot(normalizeDenseQuantSchemes(["int8", "fp8"]), 31.2);

  const held = hooks.withStableSchemes(current, next);

  assert.equal(
    held.denseQuantSchemes,
    next.denseQuantSchemes,
    "a reordered ladder names a different best scheme, so it is a real change",
  );
});

test("the first snapshot to arrive is published as-is", () => {
  const cold = snapshot([], 0);
  const warm = snapshot(normalizeDenseQuantSchemes(["fp8"]), 31.2);

  const held = hooks.withStableSchemes(cold, warm);

  assert.equal(held.denseQuantSchemes, warm.denseQuantSchemes);
  assert.deepEqual(held.denseQuantSchemes, ["fp8"]);
});
