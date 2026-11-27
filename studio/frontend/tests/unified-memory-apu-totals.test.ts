// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** An APU's GTT window may move between dedicated and shared, but the TOTAL must not:
 * fit verdicts are measured against the aggregate. */

import assert from "node:assert/strict";
import test from "node:test";
import {
  aggregateGpuMemoryTotalGb,
  gpuMemoryTotalsGb,
  gpuSharedHostMemoryGb,
  resolveMemoryCapacityGb,
  sharesHostMemory,
} from "../src/hooks/gpu-vram.ts";

/** Strix Halo gfx1151 under ROCm/Linux. host-backed is a MEASURED zero: the 64 GiB is a
 * BIOS carve-out disjoint from MemTotal (allocating 8 GiB VRAM did not move MemAvailable). */
const linuxApu = {
  memory_total_gb: 64.0,
  shared_memory: false,
  unified_memory: true,
  shared_memory_host_backed_gb: 0.0,
};
/** Split UNKNOWN (sysfs unreadable, as on WSL): absent means assume the whole window is host RAM. */
const linuxApuUnknownSplit = {
  memory_total_gb: 64.0,
  shared_memory: false,
  unified_memory: true,
};
const windowsApu = {
  memory_total_gb: 68.0,
  shared_memory: true,
  unified_memory: true,
  shared_memory_host_backed_gb: 44.0,
};
const discrete = { memory_total_gb: 179.06, shared_memory: false };

test("a Linux ROCm APU whose carve-out is MEASURED reads as dedicated VRAM", () => {
  // The measured zero means the carve-out is as dedicated as a discrete card's.
  const totals = gpuMemoryTotalsGb([linuxApu]);
  assert.equal(totals.dedicated, 64);
  assert.equal(totals.shared, 0);
  assert.equal(totals.total, 64);
  assert.equal(aggregateGpuMemoryTotalGb([linuxApu]), 64);
});

test("a Linux ROCm APU whose split is UNKNOWN is still shared host memory", () => {
  // With nothing measured, assume a view INTO system RAM; only a measured zero moves it to dedicated.
  const totals = gpuMemoryTotalsGb([linuxApuUnknownSplit]);
  assert.equal(totals.dedicated, 0);
  assert.equal(totals.shared, 64);
  assert.equal(totals.total, 64);
  assert.equal(aggregateGpuMemoryTotalGb([linuxApuUnknownSplit]), 64);
});

test("the aggregate does not move between the measured and unknown splits", () => {
  assert.equal(
    gpuMemoryTotalsGb([linuxApu]).total,
    gpuMemoryTotalsGb([linuxApuUnknownSplit]).total,
  );
});

test("a discrete card beside an APU keeps its own VRAM separate", () => {
  const totals = gpuMemoryTotalsGb([
    { memory_total_gb: 16, shared_memory: false },
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
  ]);
  assert.equal(totals.dedicated, 16);
  assert.equal(totals.shared, 48);
  assert.equal(totals.total, 64);
});

test("a multi-socket unified host keeps one pool per socket", () => {
  // `unified_memory` shares with its OWN cpu, so a multi-socket node has a pool per socket.
  const totals = gpuMemoryTotalsGb([
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
  ]);
  assert.equal(totals.dedicated, 0);
  assert.equal(totals.shared, 96);
  assert.equal(totals.total, 96);
});

test("devices that already carried the shared flag are still one pool", () => {
  // Duplicate Vulkan ICD rows for one device.
  const totals = gpuMemoryTotalsGb([
    { memory_total_gb: 48, shared_memory: true },
    { memory_total_gb: 48, shared_memory: true },
  ]);
  assert.equal(totals.shared, 48);
  assert.equal(totals.total, 48);
});

test("the aggregate survives the reclassification in every shape", () => {
  const inventories = [
    [{ memory_total_gb: 64, shared_memory: false, unified_memory: true }],
    [
      { memory_total_gb: 16, shared_memory: false },
      { memory_total_gb: 48, shared_memory: false, unified_memory: true },
    ],
    [
      { memory_total_gb: 48, shared_memory: false, unified_memory: true },
      { memory_total_gb: 48, shared_memory: false, unified_memory: true },
    ],
  ];
  for (const devices of inventories) {
    const totals = gpuMemoryTotalsGb(devices);
    const asDedicatedSum = devices.reduce(
      (sum, device) => sum + (device.memory_total_gb ?? 0),
      0,
    );
    assert.equal(totals.total, asDedicatedSum, JSON.stringify(devices));
  }
});

test("the platforms that already reported a shared pool do not move", () => {
  const totals = gpuMemoryTotalsGb([windowsApu]);
  assert.equal(totals.dedicated, 24);
  assert.equal(totals.shared, 44);
  assert.equal(totals.total, 68);
});

test("a discrete-only host is untouched", () => {
  const totals = gpuMemoryTotalsGb([discrete]);
  assert.equal(totals.dedicated, 179.06);
  assert.equal(totals.shared, 0);
  assert.equal(totals.total, 179.06);
  assert.deepEqual(gpuMemoryTotalsGb([]), { dedicated: 0, shared: 0, total: 0 });
});

test("only the two flags route; nothing else about a device does", () => {
  for (const device of [
    { memory_total_gb: 64, shared_memory: false, unified_memory: false },
    { memory_total_gb: 64, shared_memory: false },
    { memory_total_gb: 64 },
  ]) {
    assert.equal(gpuMemoryTotalsGb([device]).dedicated, 64, JSON.stringify(device));
    assert.equal(gpuMemoryTotalsGb([device]).shared, 0, JSON.stringify(device));
  }
});

test("callers that already folded the two flags get the same answer", () => {
  // Folding is NOT idempotent for several unified devices, so no caller pre-folds.
  for (const device of [linuxApu, windowsApu, discrete]) {
    const preFolded = {
      memory_total_gb: device.memory_total_gb,
      shared_memory: sharesHostMemory({
        sharedMemory: device.shared_memory === true,
        unifiedMemory: (device as { unified_memory?: boolean }).unified_memory === true,
      }),
      shared_memory_host_backed_gb: (
        device as { shared_memory_host_backed_gb?: number }
      ).shared_memory_host_backed_gb,
    };
    assert.deepEqual(gpuMemoryTotalsGb([preFolded]), gpuMemoryTotalsGb([device]));
  }
});

test("every call site agrees on a multi-socket unified host", () => {
  // The totals and capacity paths must give the same answer for multi-socket inventories.
  const devices = [
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
  ];
  assert.equal(gpuMemoryTotalsGb(devices).total, 96);
  assert.equal(gpuSharedHostMemoryGb(devices), 96);

  const pinned = devices.map((device) => ({
    memoryTotalGb: device.memory_total_gb,
    sharedMemory: device.shared_memory,
    unifiedMemory: device.unified_memory,
    sharedMemoryHostBackedGb: null,
  }));
  const capacity = resolveMemoryCapacityGb({
    pinnedDevices: pinned,
    hostDevices: pinned,
    hostGpuTotalGb: 96,
    hostDedicatedGpuTotalGb: 0,
    hostSharesSystemRam: true,
    systemRamTotalGb: 128,
    unifiedMemory: false,
  });
  assert.equal(capacity.gpuCapacityGb, 96);
});

test("one host pool is still counted once, on every path", () => {
  // `shared_memory` devices ARE views of one pool, so widening the unified case must not widen this.
  const devices = [
    { memory_total_gb: 48, shared_memory: true },
    { memory_total_gb: 48, shared_memory: true },
  ];
  assert.equal(gpuMemoryTotalsGb(devices).total, 48);
  assert.equal(gpuSharedHostMemoryGb(devices), 48);

  const pinned = devices.map((device) => ({
    memoryTotalGb: device.memory_total_gb,
    sharedMemory: device.shared_memory,
    sharedMemoryHostBackedGb: null,
  }));
  const capacity = resolveMemoryCapacityGb({
    pinnedDevices: pinned,
    hostDevices: pinned,
    hostGpuTotalGb: 48,
    hostDedicatedGpuTotalGb: 0,
    hostSharesSystemRam: true,
    systemRamTotalGb: 128,
    unifiedMemory: false,
  });
  assert.equal(capacity.gpuCapacityGb, 48);
});
