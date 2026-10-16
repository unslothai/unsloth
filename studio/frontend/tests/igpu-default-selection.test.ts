// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * 2x R9700 (32 GiB) plus a Ryzen 9950X iGPU whose 36 GiB is shared system RAM: the Train page read
 * "VRAM 100 GiB" and the chat GPU picker ticked the iGPU by default. Device rows are pinned here,
 * never read from the host.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SystemGpuDevice,
  defaultGpuIds,
  reconcileGpuSelection,
} from "../src/hooks/gpu-selection.ts";
import {
  type MemoryTotalDevice,
  aggregateGpuMemoryTotalGb,
  defaultGpuMemoryTotalGb,
} from "../src/hooks/gpu-vram.ts";

const R9700: MemoryTotalDevice = { memory_total_gb: 31.86 };
const IGPU_SHARED: MemoryTotalDevice = {
  memory_total_gb: 36.23,
  shared_memory: true,
  shared_memory_host_backed_gb: 34.26,
  unified_memory: true,
};
// Same iGPU on a stack where the driver did not mark it integrated, but torch has no gfx1036 kernels.
const IGPU_NO_KERNELS: MemoryTotalDevice = {
  memory_total_gb: 36.23,
  no_torch_kernels: true,
};

function device(
  index: number,
  overrides: Partial<SystemGpuDevice> = {},
): SystemGpuDevice {
  return {
    index,
    indexKind: "physical",
    name: `GPU ${index}`,
    memoryTotalGb: 31.86,
    memoryFreeGb: 31.86,
    sharedMemory: false,
    pinnable: true,
    diffusionPinnable: true,
    ...overrides,
  };
}

test("the default VRAM total leaves a shared-memory iGPU out", () => {
  const devices = [R9700, R9700, IGPU_SHARED];
  assert.equal(aggregateGpuMemoryTotalGb(devices), 99.95);
  assert.equal(defaultGpuMemoryTotalGb(devices), 63.72);
});

test("the default VRAM total leaves a GPU torch cannot run out", () => {
  assert.equal(defaultGpuMemoryTotalGb([R9700, R9700, IGPU_NO_KERNELS]), 63.72);
});

test("an iGPU alone, or discrete cards alone, count as before", () => {
  assert.equal(defaultGpuMemoryTotalGb([IGPU_SHARED]), 36.23);
  assert.equal(defaultGpuMemoryTotalGb([R9700, R9700]), 63.72);
  assert.equal(defaultGpuMemoryTotalGb([]), 0);
});

test("an untouched picker ticks only the discrete cards", () => {
  assert.deepEqual(
    defaultGpuIds([
      device(0),
      device(1),
      device(2, { sharedMemory: true, unifiedMemory: true }),
    ]),
    [0, 1],
  );
  assert.deepEqual(
    defaultGpuIds([device(0), device(1), device(2, { noTorchKernels: true })]),
    [0, 1],
  );
});

test("no default narrowing when there is nothing to narrow", () => {
  assert.equal(defaultGpuIds([device(0), device(1)]), null);
  assert.equal(
    defaultGpuIds([
      device(0, { sharedMemory: true }),
      device(1, { unifiedMemory: true }),
    ]),
    null,
  );
  assert.equal(defaultGpuIds([]), null);
  assert.equal(defaultGpuIds(null), null);
});

test("an Automatic selection sends the default ids, a manual pick is kept", () => {
  assert.deepEqual(
    reconcileGpuSelection(null, undefined, "physical", [0, 1, 2], [0, 1]),
    { ids: [0, 1], indexKind: "physical" },
  );
  // Ticking the iGPU by hand still reaches /load.
  assert.deepEqual(
    reconcileGpuSelection([0, 2], "physical", "physical", [0, 1, 2], [0, 1]),
    { ids: [0, 2], indexKind: "physical" },
  );
  // Without a default, or before discovery settles, Automatic stays null.
  assert.deepEqual(
    reconcileGpuSelection(null, undefined, "physical", [0, 1], null),
    { ids: null, indexKind: null },
  );
  assert.deepEqual(
    reconcileGpuSelection(null, undefined, undefined, null, [0, 1]),
    { ids: null, indexKind: null },
  );
});
