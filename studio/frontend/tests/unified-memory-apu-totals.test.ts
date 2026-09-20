// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A ROCm APU's GTT window is a view INTO system RAM, and the totals must say so.
 *
 * `hardware.py` sets `shared_memory` only on Windows, so on Linux the very same
 * part arrives as `unified_memory: true, shared_memory: false`. Splitting the
 * totals on `shared_memory` alone counted that window as dedicated VRAM standing
 * beside system RAM, which is the shape reported as an iGPU with overinflated
 * VRAM: a Strix Halo reads as a 64 GiB card, and a discrete card beside an APU
 * reads as one pool of both.
 *
 * The numbers below are measured, on a Strix Halo gfx1151 box (Radeon 8060S):
 * the ROCm inventory reports `memory_total_gb: 64.0`,`shared_memory: false`,
 * `unified_memory: true` on Linux, while the machine has 122.2 GiB of RAM and
 * the same silicon under Vulkan reports a 93.27 GiB pool.
 *
 * The split moves; the TOTAL must not. Every fit verdict is measured against the
 * aggregate, so a change there would be a change to what the app agrees to load.
 */

import assert from "node:assert/strict";
import test from "node:test";
import {
  aggregateGpuMemoryTotalGb,
  gpuMemoryTotalsGb,
  sharesHostMemory,
} from "../src/hooks/gpu-vram.ts";

/** The ROCm view of a Strix Halo APU on Linux, as measured. */
const linuxApu = {
  memory_total_gb: 64.0,
  shared_memory: false,
  unified_memory: true,
};
/** The same part on Windows, where hardware.py does set the flag. */
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
  // The split moves, the aggregate does not: fit verdicts read this figure.
  assert.equal(totals.total, 64);
  assert.equal(aggregateGpuMemoryTotalGb([linuxApu]), 64);
});

test("a discrete card beside an APU keeps its own VRAM separate", () => {
  const totals = gpuMemoryTotalsGb([
    { memory_total_gb: 16, shared_memory: false },
    { memory_total_gb: 48, shared_memory: false, unified_memory: true },
  ]);
  // Previously 64 dedicated and 0 shared, which reads as a 64 GiB card on a
  // machine whose only VRAM is the 16 GiB one.
  assert.equal(totals.dedicated, 16);
  assert.equal(totals.shared, 48);
  assert.equal(totals.total, 64);
});

test("a multi-socket unified host keeps one pool per socket", () => {
  // `shared_memory` means "this budget IS the host's own pool", so several of those
  // are views of one thing and the largest is it. `unified_memory` alone says only
  // that a device shares memory with its OWN cpu, which on a multi-socket unified
  // node is a pool per socket. Collapsing these would report such a node as a single
  // card, and it is also not what they were counted as before being classified
  // shared at all: the aggregate has to survive the reclassification.
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
  // The split is a presentation and budgeting question; the total is what fit
  // verdicts are measured against, so it has to be identical to what the old
  // dedicated-sum produced for the same inventory.
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
  // use-gpu-info and memory-fit both hand in `shared_memory: sharesHostMemory(...)`
  // and no `unified_memory` key. Folding an already-folded flag has to be a no-op,
  // or this change would move the paths that had learned the rule on their own.
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
