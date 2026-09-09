// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { SystemGpuDevice } from "../src/hooks/gpu-selection.ts";
import { aggregateVramReserveDeficitGb } from "../src/hooks/gpu-vram.ts";
import {
  type MemoryFitCapacity,
  resolveMemoryFit,
  resolveReclaimableMemoryCredit,
} from "../src/features/model-picker/model-config/memory-fit.ts";

const GiB = 1024 ** 3;
const pool = { ids: null, indexKind: null };
const estimate = (total: number, gpu = total, files = 0) => ({
  totalBytes: total * GiB,
  gpuBytes: gpu * GiB,
  weightsBytes: files * GiB,
  kvEstimable: true,
  drafterKvUnsized: false,
  adaptersUnsized: false,
  moeOffloadUnmodelled: false,
});
const capacity: MemoryFitCapacity = {
  gpuCapacityGb: 48,
  totalCapacityGb: 112,
  systemRamCapacityGb: 64,
  freeGpuCapacityGb: 24,
  usableSystemRamGb: 24,
  singleMemoryPool: false,
};

test("reload credit preserves pressure thresholds and physical limits", () => {
  assert.equal(resolveMemoryFit(estimate(25), capacity).freeGpuFit, "exceeds");
  for (const singleMemoryPool of [false, true]) {
    const available = {
      ...capacity,
      singleMemoryPool,
      ...(singleMemoryPool ? { gpuCapacityGb: 64, totalCapacityGb: 64 } : {}),
      reclaimableTotalBytes: (singleMemoryPool ? 10 : 20) * GiB,
      reclaimableGpuBytes: 10 * GiB,
    };
    for (const [wanted, verdict] of [
      [25, "fits"],
      [29, "tight"],
      [35, "exceeds"],
    ] as const) {
      const footprint = singleMemoryPool
        ? estimate(wanted, wanted / 2)
        : estimate(wanted * 2, wanted);
      const fit = resolveMemoryFit(footprint, available);
      assert.equal(fit.freeGpuFit, verdict);
      assert.equal(fit.usableHostFit, verdict);
    }
    const oversized = resolveMemoryFit(estimate(120, 70), available);
    assert.equal(oversized.rawGpuFit, "exceeds");
    assert.equal(oversized.totalFit, "exceeds");
  }
});

test("resident credit cannot turn an unknown probe into a known one", () => {
  for (const known of [false, true]) {
    const fit = resolveMemoryFit(estimate(9), {
      ...capacity,
      singleMemoryPool: true,
      freeGpuCapacityGb: 0,
      usableSystemRamGb: 0,
      freeGpuCapacityKnown: known,
      usableSystemRamKnown: known,
      reclaimableTotalBytes: 10 * GiB,
    });
    assert.equal(fit.freeGpuFit, known ? "tight" : "unknown");
    assert.equal(fit.usableHostFit, known ? "tight" : "unknown");
  }
});

test("credit fills unmet reserves without consuming already usable memory", () => {
  const deficit = aggregateVramReserveDeficitGb(
    [{ memoryTotalGb: 24, memoryFreeGb: 0 }],
    0.9,
  );
  assert.ok(Math.abs(deficit - 2.4) < 1e-9);
  const fit = resolveMemoryFit(estimate(32, 16), {
    ...capacity,
    freeGpuCapacityGb: 0,
    usableSystemRamGb: 0,
    freeGpuCapacityKnown: true,
    usableSystemRamKnown: true,
    freeGpuReserveDeficitGb: deficit,
    systemRamReserveDeficitGb: 2,
    reclaimableTotalBytes: 40 * GiB,
    reclaimableGpuBytes: 20 * GiB,
  });
  assert.equal(fit.freeGpuFit, "tight");
  assert.equal(fit.usableHostFit, "tight");
  assert.equal(
    resolveMemoryFit(estimate(8), {
      ...capacity,
      freeGpuCapacityGb: 10,
      freeGpuReserveDeficitGb: deficit,
      reclaimableTotalBytes: GiB,
      reclaimableGpuBytes: GiB,
    }).freeGpuFit,
    "fits",
  );
});

test("GPU credit requires the complete resident pool in the same namespace", () => {
  for (const [loaded, requested, gpu] of [
    [[0], [1], 0],
    [[0, 1], [1], 0],
    [[0], [0, 1], 20],
    [null, [0], 0],
    [null, null, 20],
  ] as const) {
    const credit = resolveReclaimableMemoryCredit(
      estimate(40, 20),
      { ids: loaded && [...loaded], indexKind: "physical" },
      { ids: requested && [...requested], indexKind: "physical" },
      { gpuPlacementKnown: true },
    );
    assert.deepEqual(credit, {
      totalBytes: (20 + gpu) * GiB,
      gpuBytes: gpu * GiB,
    });
  }
  const wrongNamespace = resolveReclaimableMemoryCredit(
    estimate(40, 20),
    { ids: [0], indexKind: "physical" },
    { ids: [0], indexKind: "vulkan" },
    { gpuPlacementKnown: true },
  );
  assert.deepEqual(wrongNamespace, { totalBytes: 20 * GiB, gpuBytes: 0 });
});

const sharedDevice: SystemGpuDevice = {
  index: 0,
  indexKind: "vulkan",
  name: "Shared GPU",
  memoryTotalGb: 32,
  memoryFreeGb: 0,
  sharedMemory: true,
  pinnable: true,
  diffusionPinnable: false,
};

test("only guaranteed host-backed GPU allocations return to RAM after a GPU change", () => {
  const cases: [SystemGpuDevice[], number][] = [
    [[sharedDevice], 24],
    [[{ ...sharedDevice, sharedMemory: false, unifiedMemory: true }], 24],
    [[{ ...sharedDevice, sharedMemoryHostBackedGb: 24 }], 16],
    [[{ ...sharedDevice, sharedMemory: false }], 4],
    [[], 4],
  ];
  for (const [devices, total] of cases) {
    const credit = resolveReclaimableMemoryCredit(
      estimate(24, 20),
      { ids: [0], indexKind: "vulkan" },
      { ids: [1], indexKind: "vulkan" },
      { devices, gpuPlacementKnown: true },
    );
    assert.deepEqual(credit, { totalBytes: total * GiB, gpuBytes: 0 });
  }
});

test("fitted placement and fallback allocations earn no uncertain VRAM credit", () => {
  for (const options of [{}, { gpuPlacementKnown: true, cpuFallback: true }]) {
    assert.deepEqual(
      resolveReclaimableMemoryCredit(estimate(30, 24, 4), pool, pool, options),
      { totalBytes: 2 * GiB, gpuBytes: 0 },
    );
  }
  const experts = resolveReclaimableMemoryCredit(
    { ...estimate(30, 24, 4), moeOffloadUnmodelled: true },
    pool,
    pool,
    { gpuPlacementKnown: true },
  );
  assert.deepEqual(experts, { totalBytes: 2 * GiB, gpuBytes: 0 });
});

test("mapped weights are excluded from host credit, including unified memory", () => {
  const cpu = resolveReclaimableMemoryCredit(estimate(30, 0, 22), pool, pool);
  assert.deepEqual(cpu, { totalBytes: 8 * GiB, gpuBytes: 0 });
  const unified = resolveReclaimableMemoryCredit(
    estimate(30, 30, 22),
    pool,
    pool,
    { appleUnifiedMemory: true },
  );
  assert.deepEqual(unified, cpu);
  const discrete = resolveReclaimableMemoryCredit(
    estimate(30, 20, 22),
    pool,
    pool,
    {
      gpuPlacementKnown: true,
      devices: [{ ...sharedDevice, sharedMemory: false }],
    },
  );
  assert.deepEqual(discrete, { totalBytes: 20 * GiB, gpuBytes: 20 * GiB });
});

test("duplicate shared GPU views consume one reserve deficit", () => {
  const shared = { memoryTotalGb: 32, memoryFreeGb: 0, sharedMemory: true };
  assert.ok(
    Math.abs(aggregateVramReserveDeficitGb([shared, shared], 0.9) - 3.2) < 1e-9,
  );
  assert.ok(
    Math.abs(
      aggregateVramReserveDeficitGb(
        [
          shared,
          { ...shared, memoryFreeGb: 1 },
          { ...shared, sharedMemory: false },
        ],
        0.9,
      ) - 5.4,
    ) < 1e-9,
  );
  assert.equal(
    aggregateVramReserveDeficitGb(
      [shared, { ...shared, memoryFreeGb: 4 }],
      0.9,
    ),
    0,
  );
});
