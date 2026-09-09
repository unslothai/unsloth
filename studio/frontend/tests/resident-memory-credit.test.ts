// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Load-Model memory row's verdicts and the single note it prints.
//
// This file exists because the chain used to live inside model-config-page.tsx, where
// the node runner cannot reach it: a 3,400-line .tsx that imports React, the router
// and thirty component barrels. An arm that could never be taken shipped in it, and
// nothing could have caught that except reading the ternary carefully enough.

import assert from "node:assert/strict";
import test from "node:test";
import type { SystemGpuDevice } from "../src/hooks/gpu-selection.ts";
import { aggregateVramReserveDeficitGb } from "../src/hooks/gpu-vram.ts";

import {
  type MemoryFitCapacity,
  type MemoryFitEstimate,
  resolveMemoryFit,
  resolveReclaimableMemoryCredit,
} from "../src/features/model-picker/model-config/memory-fit.ts";

const GB = 1024 ** 3;

const SIZED: MemoryFitEstimate = {
  gpuBytes: 0,
  totalBytes: 0,
  kvEstimable: true,
  drafterKvUnsized: false,
  adaptersUnsized: false,
  moeOffloadUnmodelled: false,
};

/** A roomy discrete host: 24 GB card, 64 GB of RAM, nothing else running. */
const IDLE_DISCRETE: MemoryFitCapacity = {
  gpuCapacityGb: 24,
  totalCapacityGb: 88,
  systemRamCapacityGb: 64,
  freeGpuCapacityGb: 23,
  usableSystemRamGb: 60,
  singleMemoryPool: false,
};

/** Apple, 64 GB unified. One pool for everything. */
const APPLE: MemoryFitCapacity = {
  gpuCapacityGb: 64,
  totalCapacityGb: 64,
  systemRamCapacityGb: 64,
  freeGpuCapacityGb: 60,
  usableSystemRamGb: 60,
  singleMemoryPool: true,
};

const fit = (
  estimate: Partial<MemoryFitEstimate>,
  capacity: Partial<MemoryFitCapacity>,
  base: MemoryFitCapacity = IDLE_DISCRETE,
) => resolveMemoryFit({ ...SIZED, ...estimate }, { ...base, ...capacity });

test("reloading an unchanged config does not warn about the memory it is about to release", () => {
  // Apple 64 GB, a 25.61 GiB load, 25.53 GiB free because that same model is holding the rest.
  // Uncredited this is ratio 1.003 -> exceeds.
  const uncredited = fit(
    { gpuBytes: 25.61 * GB, totalBytes: 25.61 * GB },
    { freeGpuCapacityGb: 25.53, usableSystemRamGb: 25.53 },
    APPLE,
  );
  assert.equal(uncredited.usableHostFit, "exceeds");
  assert.ok(uncredited.advisory);

  const credited = fit(
    { gpuBytes: 25.61 * GB, totalBytes: 25.61 * GB },
    {
      freeGpuCapacityGb: 25.53,
      usableSystemRamGb: 25.53,
      reclaimableTotalBytes: 25.61 * GB,
      reclaimableGpuBytes: 25.61 * GB,
    },
    APPLE,
  );
  assert.equal(credited.gpuPressured, false);
  assert.equal(credited.hostPressured, false);
  assert.equal(credited.advisory, null);
});

test("growing the context past what the resident copy returns still warns", () => {
  // 40 GiB wanted, with 12 GiB free and 25 GiB returning on unload.
  const result = fit(
    { gpuBytes: 40 * GB, totalBytes: 40 * GB },
    {
      freeGpuCapacityGb: 12,
      usableSystemRamGb: 12,
      reclaimableTotalBytes: 25 * GB,
      reclaimableGpuBytes: 25 * GB,
    },
    APPLE,
  );
  assert.ok(result.gpuPressured || result.hostPressured);
  assert.ok(result.advisory);
});

test("reclaimed memory preserves pressure thresholds in every pool", () => {
  for (const [wanted, expected] of [
    [34, "fits"],
    [35, "tight"],
    [40, "tight"],
    [41, "exceeds"],
  ] as const) {
    const gpu = fit(
      { gpuBytes: wanted * GB, totalBytes: wanted * GB },
      {
        freeGpuCapacityGb: 30,
        reclaimableTotalBytes: 10 * GB,
        reclaimableGpuBytes: 10 * GB,
      },
    );
    assert.equal(gpu.freeGpuFit, expected);
    const host = fit(
      { gpuBytes: 0, totalBytes: wanted * GB },
      {
        usableSystemRamGb: 30,
        reclaimableTotalBytes: 10 * GB,
      },
    );
    assert.equal(host.usableHostFit, expected);
    const shared = fit(
      { gpuBytes: wanted * GB, totalBytes: wanted * GB },
      {
        freeGpuCapacityGb: 30,
        usableSystemRamGb: 30,
        reclaimableTotalBytes: 10 * GB,
        reclaimableGpuBytes: 10 * GB,
      },
      APPLE,
    );
    assert.equal(shared.freeGpuFit, expected);
    assert.equal(shared.usableHostFit, expected);
  }
});

test("reclaimed memory does not turn missing free-memory readings into known ones", () => {
  for (const available of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const result = fit(
      { gpuBytes: 9 * GB, totalBytes: 9 * GB },
      {
        freeGpuCapacityGb: available,
        usableSystemRamGb: available,
        reclaimableTotalBytes: 10 * GB,
        reclaimableGpuBytes: 10 * GB,
      },
      APPLE,
    );
    assert.equal(result.freeGpuFit, "unknown");
    assert.equal(result.usableHostFit, "unknown");
  }
  const exhausted = fit(
    { gpuBytes: 9 * GB, totalBytes: 9 * GB },
    {
      freeGpuCapacityGb: 0,
      usableSystemRamGb: 0,
      freeGpuCapacityKnown: true,
      usableSystemRamKnown: true,
      reclaimableTotalBytes: 10 * GB,
      reclaimableGpuBytes: 10 * GB,
    },
    APPLE,
  );
  assert.equal(exhausted.freeGpuFit, "tight");
  assert.equal(exhausted.usableHostFit, "tight");
});

test("reclaimed memory first fills GPU and host reserve deficits", () => {
  const gpuDeficit = aggregateVramReserveDeficitGb(
    [{ memoryTotalGb: 24, memoryFreeGb: 0 }],
    0.9,
  );
  assert.ok(Math.abs(gpuDeficit - 2.4) < 1e-9);
  const result = fit(
    { gpuBytes: 16 * GB, totalBytes: 32 * GB },
    {
      freeGpuCapacityGb: 0,
      freeGpuCapacityKnown: true,
      freeGpuReserveDeficitGb: gpuDeficit,
      usableSystemRamGb: 0,
      usableSystemRamKnown: true,
      systemRamReserveDeficitGb: 2,
      reclaimableTotalBytes: 40 * GB,
      reclaimableGpuBytes: 20 * GB,
    },
  );
  assert.equal(result.freeGpuFit, "tight");
  assert.equal(result.usableHostFit, "tight");
});

test("unmet reserves do not reduce memory already usable on another card", () => {
  for (const credit of [0, 1]) {
    const result = fit(
      { gpuBytes: 8 * GB, totalBytes: 8 * GB },
      {
        freeGpuCapacityGb: 10,
        freeGpuReserveDeficitGb: 2.4,
        reclaimableTotalBytes: credit * GB,
        reclaimableGpuBytes: credit * GB,
      },
    );
    assert.equal(result.freeGpuFit, "fits");
  }
  assert.equal(
    aggregateVramReserveDeficitGb(
      [{ memoryTotalGb: 24, memoryFreeGb: 5 }],
      0.9,
    ),
    0,
  );
});

test("the credit never improves a CAPACITY verdict, only a free-memory one", () => {
  // 70 GB on a 64 GB machine: unloading frees memory, it does not add any.
  const result = fit(
    { gpuBytes: 70 * GB, totalBytes: 70 * GB },
    {
      freeGpuCapacityGb: 2,
      usableSystemRamGb: 2,
      reclaimableTotalBytes: 70 * GB,
      reclaimableGpuBytes: 70 * GB,
    },
    APPLE,
  );
  assert.equal(result.totalFit, "exceeds");
  assert.equal(result.advisory?.tone, "warn");
  assert.ok(result.advisory);
});

test("on discrete memory the credit follows the pool each byte came from", () => {
  // 20 GB on the card, 20 GB on the host. The host credit is the part that was NOT on the GPU.
  const result = fit(
    { gpuBytes: 20 * GB, totalBytes: 40 * GB },
    {
      freeGpuCapacityGb: 4,
      usableSystemRamGb: 4,
      reclaimableTotalBytes: 40 * GB,
      reclaimableGpuBytes: 20 * GB,
    },
  );
  assert.equal(result.gpuPressured, false);
  assert.equal(result.hostPressured, false);
});

test("resident GPU credit requires the whole loaded pool to remain available", () => {
  const resident = { gpuBytes: 20 * GB, totalBytes: 40 * GB };
  for (const [loaded, requested, credited] of [
    [[0], [1], false],
    [[0, 1], [1], false],
    [[0], [0, 1], true],
    [[0, 1], [1, 0], true],
    [null, [0], false],
    [[0], null, true],
    [null, null, true],
  ] as const) {
    const credit = resolveReclaimableMemoryCredit(
      { weightsBytes: 0, ...resident },
      { ids: loaded ? [...loaded] : null, indexKind: "physical" },
      { ids: requested ? [...requested] : null, indexKind: "physical" },
      { cpuFallback: false, devices: [], gpuPlacementKnown: true },
    );
    const result = fit(resident, {
      freeGpuCapacityGb: 4,
      usableSystemRamGb: 4,
      reclaimableTotalBytes: credit.totalBytes,
      reclaimableGpuBytes: credit.gpuBytes,
    });
    assert.equal(result.gpuPressured, !credited);
    assert.equal(result.hostPressured, false);
    assert.equal(credit.totalBytes - credit.gpuBytes, 20 * GB);
  }
});

test("resident GPU IDs from another namespace do not earn VRAM credit", () => {
  const credit = resolveReclaimableMemoryCredit(
    { weightsBytes: 0, ...{ gpuBytes: 20 * GB, totalBytes: 40 * GB } },
    { ids: [0], indexKind: "physical" },
    { ids: [0], indexKind: "vulkan" },
    { cpuFallback: false, devices: [], gpuPlacementKnown: true },
  );
  assert.deepEqual(credit, { gpuBytes: 0, totalBytes: 20 * GB });
});

test("excluded VRAM credit never becomes extra RAM credit", () => {
  const credit = resolveReclaimableMemoryCredit(
    { weightsBytes: 0, ...{ gpuBytes: 20 * GB, totalBytes: 40 * GB } },
    { ids: [0], indexKind: "physical" },
    { ids: [1], indexKind: "physical" },
    { cpuFallback: false, devices: [], gpuPlacementKnown: true },
  );
  const result = fit(
    { gpuBytes: 20 * GB, totalBytes: 50 * GB },
    {
      freeGpuCapacityGb: 4,
      usableSystemRamGb: 4,
      reclaimableTotalBytes: credit.totalBytes,
      reclaimableGpuBytes: credit.gpuBytes,
    },
  );
  assert.equal(result.gpuPressured, true);
  assert.equal(result.hostPressured, true);
});

test("CPU fallback and unmodelled experts withhold uncertain VRAM credit", () => {
  const pool = { ids: null, indexKind: null };
  for (const cpuFallback of [false, true]) {
    const credit = resolveReclaimableMemoryCredit(
      {
        weightsBytes: 0,
        ...{
          gpuBytes: 20 * GB,
          totalBytes: 40 * GB,
          moeOffloadUnmodelled: !cpuFallback,
        },
      },
      pool,
      pool,
      { cpuFallback: cpuFallback, devices: [], gpuPlacementKnown: true },
    );
    assert.deepEqual(credit, { gpuBytes: 0, totalBytes: 20 * GB });
    const result = fit(
      { gpuBytes: 20 * GB, totalBytes: 30 * GB },
      {
        freeGpuCapacityGb: 4,
        usableSystemRamGb: 4,
        reclaimableTotalBytes: credit.totalBytes,
        reclaimableGpuBytes: credit.gpuBytes,
      },
    );
    assert.equal(result.gpuPressured, true);
    assert.equal(result.hostPressured, false);
  }
});

test("shared GPU allocations return to RAM when the requested GPU changes", () => {
  const device: SystemGpuDevice = {
    index: 0,
    indexKind: "vulkan",
    name: "Shared GPU",
    memoryTotalGb: 32,
    memoryFreeGb: 0,
    sharedMemory: true,
    pinnable: true,
    diffusionPinnable: false,
  };
  const resident = { gpuBytes: 20 * GB, totalBytes: 24 * GB };
  const loaded = { ids: [0], indexKind: "vulkan" as const };
  const requested = { ids: [1], indexKind: "vulkan" as const };
  for (const topology of [
    device,
    { ...device, sharedMemory: false, unifiedMemory: true },
  ]) {
    const credit = resolveReclaimableMemoryCredit(
      { weightsBytes: 0, ...resident },
      loaded,
      requested,
      { cpuFallback: false, devices: [topology], gpuPlacementKnown: true },
    );
    assert.deepEqual(credit, { gpuBytes: 0, totalBytes: 24 * GB });
    const result = fit(
      { gpuBytes: 8 * GB, totalBytes: 28 * GB },
      {
        usableSystemRamGb: 4,
        reclaimableTotalBytes: credit.totalBytes,
        reclaimableGpuBytes: credit.gpuBytes,
      },
    );
    assert.equal(result.hostPressured, false);
  }
  const partial = resolveReclaimableMemoryCredit(
    { weightsBytes: 0, ...resident },
    loaded,
    requested,
    {
      cpuFallback: false,
      devices: [{ ...device, sharedMemoryHostBackedGb: 24 }],
      gpuPlacementKnown: true,
    },
  );
  assert.deepEqual(partial, { gpuBytes: 0, totalBytes: 16 * GB });
  const unknown = resolveReclaimableMemoryCredit(
    { weightsBytes: 0, ...resident },
    loaded,
    requested,
    { cpuFallback: false, devices: [], gpuPlacementKnown: true },
  );
  assert.deepEqual(unknown, { gpuBytes: 0, totalBytes: 4 * GB });
  const discrete = resolveReclaimableMemoryCredit(
    { weightsBytes: 0, ...resident },
    loaded,
    requested,
    {
      cpuFallback: false,
      devices: [{ ...device, sharedMemory: false }],
      gpuPlacementKnown: true,
    },
  );
  assert.deepEqual(discrete, unknown);
});

test("fitted resident placement cannot supply an estimated VRAM credit", () => {
  const pool = { ids: null, indexKind: null };
  const resident = {
    totalBytes: 30 * GB,
    gpuBytes: 24 * GB,
    weightsBytes: 4 * GB,
  };
  const credit = resolveReclaimableMemoryCredit(resident, pool, pool);
  assert.deepEqual(credit, { totalBytes: 2 * GB, gpuBytes: 0 });
  const result = fit(
    { totalBytes: 20 * GB, gpuBytes: 18 * GB },
    {
      freeGpuCapacityGb: 4,
      reclaimableTotalBytes: credit.totalBytes,
      reclaimableGpuBytes: credit.gpuBytes,
    },
  );
  assert.equal(result.freeGpuFit, "exceeds");
  const shared = resolveReclaimableMemoryCredit(resident, pool, pool, {
    cpuFallback: false,
    devices: [],
    gpuPlacementKnown: false,
    appleUnifiedMemory: true,
  });
  assert.deepEqual(shared, { totalBytes: 22 * GB, gpuBytes: 0 });
});

test("resident host credit excludes potentially reclaimable mapped files", () => {
  const pool = { ids: null, indexKind: null };
  for (const [gpu, total, files, hostCredit] of [
    [0, 30, 22, 8],
    [20, 30, 22, 0],
  ]) {
    const credit = resolveReclaimableMemoryCredit(
      {
        totalBytes: total * GB,
        gpuBytes: gpu * GB,
        weightsBytes: files * GB,
      },
      pool,
      pool,
      { cpuFallback: false, devices: [], gpuPlacementKnown: true },
    );
    assert.equal(credit.totalBytes - credit.gpuBytes, hostCredit * GB);
  }
  const shared = resolveReclaimableMemoryCredit(
    {
      totalBytes: 30 * GB,
      gpuBytes: 30 * GB,
      weightsBytes: 22 * GB,
    },
    pool,
    pool,
    {
      cpuFallback: false,
      devices: [],
      gpuPlacementKnown: false,
      appleUnifiedMemory: true,
    },
  );
  assert.deepEqual(shared, { totalBytes: 8 * GB, gpuBytes: 0 });
  for (const weightsBytes of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
    assert.deepEqual(
      resolveReclaimableMemoryCredit(
        {
          totalBytes: 30 * GB,
          gpuBytes: 20 * GB,
          weightsBytes,
        },
        pool,
        pool,
        { cpuFallback: false, devices: [], gpuPlacementKnown: true },
      ),
      { totalBytes: 0, gpuBytes: 0 },
    );
  }
});

test("duplicate shared GPU views consume one reserve deficit", () => {
  const shared = { memoryTotalGb: 32, memoryFreeGb: 0, sharedMemory: true };
  assert.ok(
    Math.abs(aggregateVramReserveDeficitGb([shared, shared], 0.9) - 3.2) < 1e-9,
  );
  const lowerDeficit = { ...shared, memoryFreeGb: 1 };
  const dedicated = { ...shared, sharedMemory: false };
  assert.ok(
    Math.abs(
      aggregateVramReserveDeficitGb([shared, lowerDeficit, dedicated], 0.9) -
        5.4,
    ) < 1e-9,
  );
  assert.equal(
    aggregateVramReserveDeficitGb(
      [shared, { ...shared, memoryFreeGb: 4 }],
      0.9,
    ),
    0,
  );
  const result = fit(
    { totalBytes: 15 * GB, gpuBytes: 15 * GB },
    {
      freeGpuCapacityGb: 0,
      freeGpuCapacityKnown: true,
      reclaimableTotalBytes: 22 * GB,
      reclaimableGpuBytes: 22 * GB,
      freeGpuReserveDeficitGb: aggregateVramReserveDeficitGb(
        [shared, shared],
        0.9,
      ),
    },
  );
  assert.equal(result.freeGpuFit, "fits");
});

test("a GPU credit larger than its own total cannot inflate the host share", () => {
  // GPU share above its own total. Clamping keeps the host credit at zero instead of negative.
  const result = fit(
    { gpuBytes: 20 * GB, totalBytes: 40 * GB },
    {
      freeGpuCapacityGb: 23,
      usableSystemRamGb: 60,
      reclaimableTotalBytes: 10 * GB,
      reclaimableGpuBytes: 999 * GB,
    },
  );
  // 20 GB host share less a credit floored at zero, against 60 GB usable.
  assert.equal(result.hostPressured, false);
  assert.ok(Number.isFinite(result.hostShareBytes));
});

test("a garbage credit cannot increase available memory", () => {
  for (const credit of [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    -50 * GB,
    undefined,
  ]) {
    const result = fit(
      { gpuBytes: 30 * GB, totalBytes: 30 * GB },
      {
        freeGpuCapacityGb: 6,
        usableSystemRamGb: 6,
        reclaimableTotalBytes: credit,
        reclaimableGpuBytes: credit,
      },
      APPLE,
    );
    // The warning a real 30 GB load under 6 GB free deserves, unchanged.
    assert.ok(
      result.gpuPressured || result.hostPressured,
      `credit ${String(credit)} should not have silenced the warning`,
    );
  }
});

test("an unmeasurable footprint stays unknown with reclaimed capacity", () => {
  const result = fit(
    { gpuBytes: Number.NaN, totalBytes: Number.NaN },
    {
      freeGpuCapacityGb: 6,
      usableSystemRamGb: 6,
      reclaimableTotalBytes: 30 * GB,
      reclaimableGpuBytes: 30 * GB,
    },
    APPLE,
  );
  assert.equal(result.freeGpuFit, "unknown");
  assert.equal(result.usableHostFit, "unknown");
});
