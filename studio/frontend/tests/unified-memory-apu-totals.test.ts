// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A Linux ROCm APU's GTT window was counted as dedicated VRAM standing beside the
 * system RAM it is a view into. The split moves; the TOTAL must not, since fit
 * verdicts are measured against the aggregate.
 */

import assert from "node:assert/strict";
import test from "node:test";
import {
  aggregateGpuMemoryTotalGb,
  gpuMemoryTotalsGb,
  gpuSharedHostMemoryGb,
  resolveMemoryCapacityGb,
  sharesHostMemory,
} from "../src/hooks/gpu-vram.ts";

/** Measured on a Strix Halo gfx1151 box (Radeon 8060S) under ROCm on Linux. */
const linuxApu = {
  memory_total_gb: 64.0,
  shared_memory: false,
  unified_memory: true,
};
/** The same part on Windows, where `hardware.py` does set the flag. */
const windowsApu = {
  memory_total_gb: 68.0,
  shared_memory: true,
  unified_memory: true,
  shared_memory_host_backed_gb: 44.0,
};
const discrete = { memory_total_gb: 179.06, shared_memory: false };

test("a Linux ROCm APU's window is shared host memory, not dedicated VRAM", () => {
  const totals = gpuMemoryTotalsGb([linuxApu]);
  assert.equal(totals.dedicated, 0);
  assert.equal(totals.shared, 64);
  assert.equal(totals.total, 64);
  assert.equal(aggregateGpuMemoryTotalGb([linuxApu]), 64);
});

test("a discrete card beside an APU keeps its own VRAM separate", () => {
  const totals = gpuMemoryTotalsGb([
    { memory_total_gb: 16, shared_memory: false },
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
  ]);
  // Previously 64 dedicated: a 64 GiB card on a machine whose only VRAM is 16 GiB.
  assert.equal(totals.dedicated, 16);
  assert.equal(totals.shared, 48);
  assert.equal(totals.total, 64);
});

test("a multi-socket unified host keeps one pool per socket", () => {
  // `unified_memory` alone says a device shares memory with its OWN cpu, which on a
  // multi-socket node is a pool per socket: collapsing these reports it as one card.
  const totals = gpuMemoryTotalsGb([
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
  ]);
  assert.equal(totals.dedicated, 0);
  assert.equal(totals.shared, 96);
  assert.equal(totals.total, 96);
});

test("devices that already carried the shared flag are still one pool", () => {
  // The Vulkan case the max was written for: duplicate ICD rows for one device.
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
  // use-gpu-info and memory-fit hand in an already-folded flag: that has to be a no-op.
  // Single device only. Folding is NOT idempotent for several unified devices, which is
  // why no caller pre-folds any more; `every call site agrees` below pins that.
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
  // Reported by ItsRoy69 on #11366: the totals path read these as 96 while the capacity
  // path pre-folded both flags into `shared_memory` and then collapsed them to 48, so the
  // same inventory had two answers depending on who asked. A pin covering both sockets
  // reported half its memory.
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
  // The other half of the same rule: `shared_memory` devices ARE views of one pool
  // (duplicate Vulkan ICD rows), so widening the unified case must not widen this one.
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
