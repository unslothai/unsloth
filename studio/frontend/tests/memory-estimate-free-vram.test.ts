// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the Load-Model memory row measures FREE memory against, and how both the free
// and the total aggregate behave on inventories nobody has on the desk.
//
// The row's sibling file memory-estimate-capacity.test.ts covers the totals. This one
// covers the free figure that feeds `freeGpuFit`, plus the readings that arrive
// missing, null, negative or non-finite from a probe that failed halfway.

import assert from "node:assert/strict";
import test from "node:test";

import { classifyMemoryFit } from "../src/features/model-picker/model-config/memory-fit.ts";
import {
  aggregateGpuMemoryTotalGb,
  aggregateUsableFreeVramGb,
  resolveMemoryCapacityGb,
  usableFreeVramGb,
} from "../src/hooks/gpu-vram.ts";

const GB = 1024 ** 3;

// studio/backend/tests/test_system_vulkan_gpu_info.py: an RX 9070 XT with its own
// 16 GB beside an 8060S iGPU. get_vulkan_inference_gpu_info forces the iGPU's
// total_mib to 0 and reports `budget_mib = total_mib or free_mib`, so an iGPU's
// memory_total_gb and vram_free_gb are the SAME number: the host's free RAM, less the
// iGPU host reserve.
const DGPU = { memoryFreeGb: 15, memoryTotalGb: 16, sharedMemory: false };
const IGPU = { memoryFreeGb: 11, memoryTotalGb: 11, sharedMemory: true };

// ---------------------------------------------------------------------------
// D3: the shared pool, counted once

test("one discrete card and one iGPU are added, exactly as their totals are", () => {
  // Recorded deliberately: this pairing is NOT the double-count case.
  // aggregateGpuMemoryTotalGb adds these two as well, so the free figure and the
  // total it is weighed against agree. Taking max() over a single shared device is
  // that device.
  assert.equal(
    aggregateGpuMemoryTotalGb([
      { memory_total_gb: 16, shared_memory: false },
      { memory_total_gb: 11, shared_memory: true },
    ]),
    27,
  );
  const free = aggregateUsableFreeVramGb([DGPU, IGPU], 1);
  const summedBlindly =
    usableFreeVramGb(15, 16, 1) + usableFreeVramGb(11, 11, 1);
  assert.ok(Math.abs(free - summedBlindly) < 0.01);
});

test("the SAME pool reported twice does not double the free figure", () => {
  // ggml-vulkan enumerates one device per installed ICD, and _run_vulkan_probe does
  // not dedup, so a box with both Mesa RADV and AMDVLK present lists one physical
  // iGPU at two ordinals, each reporting the same host RAM. The totals aggregate
  // already takes max() over shared devices; a plain sum of the free readings did
  // not, and invented a pool the machine does not have.
  const once = aggregateUsableFreeVramGb([DGPU, IGPU], 1);
  const twice = aggregateUsableFreeVramGb([DGPU, IGPU, IGPU], 1);
  assert.equal(twice, once);
  // Four ICDs would have quadrupled it.
  assert.equal(aggregateUsableFreeVramGb([DGPU, IGPU, IGPU, IGPU, IGPU], 1), once);
  // And the total says the same thing, so the two figures cannot disagree.
  assert.equal(
    aggregateGpuMemoryTotalGb([
      { memory_total_gb: 16, shared_memory: false },
      { memory_total_gb: 11, shared_memory: true },
      { memory_total_gb: 11, shared_memory: true },
    ]),
    27,
  );
});

test("the phantom capacity was enough to flip the verdict", () => {
  const blind = [DGPU, IGPU, IGPU].reduce(
    (sum, d) => sum + usableFreeVramGb(d.memoryFreeGb, d.memoryTotalGb, 1),
    0,
  );
  const correct = aggregateUsableFreeVramGb([DGPU, IGPU, IGPU], 1);
  assert.ok(blind - correct > 10, `phantom was ${(blind - correct).toFixed(2)} GB`);
  assert.equal(classifyMemoryFit(30 * GB, blind), "fits");
  assert.equal(classifyMemoryFit(30 * GB, correct), "exceeds");
});

test("several discrete cards still sum, because they are separate pools", () => {
  const pair = aggregateUsableFreeVramGb(
    [
      { memoryFreeGb: 24, memoryTotalGb: 24, sharedMemory: false },
      { memoryFreeGb: 24, memoryTotalGb: 24, sharedMemory: false },
    ],
    1,
  );
  assert.ok(Math.abs(pair - 2 * usableFreeVramGb(24, 24, 1)) < 0.01);
});

test("two shared devices of different sizes report the larger, not their sum", () => {
  // Two views of one pool, one of which had a reserve applied and one of which did
  // not. The pool is at least the larger; it is certainly not both added together.
  const mixed = aggregateUsableFreeVramGb(
    [
      { memoryFreeGb: 40, memoryTotalGb: 40, sharedMemory: true },
      { memoryFreeGb: 11, memoryTotalGb: 11, sharedMemory: true },
    ],
    1,
  );
  assert.ok(Math.abs(mixed - usableFreeVramGb(40, 40, 1)) < 0.01);
});

test("partial APUs retain each reserved segment beside one shared pool", () => {
  const apu = {
    memoryFreeGb: 100,
    memoryTotalGb: 100,
    sharedMemory: true,
    sharedMemoryHostBackedGb: 92,
  };
  assert.equal(aggregateUsableFreeVramGb([apu], 0.8), 80);
  assert.equal(aggregateUsableFreeVramGb([apu, apu], 0.8), 108);
});

test("busy partial APUs do not repeat the currently free shared pool", () => {
  const apu = {
    memoryFreeGb: 50,
    memoryTotalGb: 100,
    sharedMemory: true,
    sharedMemoryHostBackedGb: 92,
  };
  assert.equal(aggregateUsableFreeVramGb([apu, apu], 0.8), 50);
});

test("a smaller partial APU does not cap a larger shared aperture", () => {
  const large = {
    memoryFreeGb: 100,
    memoryTotalGb: 100,
    sharedMemory: true,
    sharedMemoryHostBackedGb: 92,
  };
  const small = {
    memoryFreeGb: 20,
    memoryTotalGb: 20,
    sharedMemory: true,
    sharedMemoryHostBackedGb: 12,
  };
  assert.equal(aggregateUsableFreeVramGb([large, small], 0.8), 96);
});

test("the VRAM Budget reserve is applied per device inside the aggregate", () => {
  // _select_gpus' own example: a 24 GB card with 10 GB free at an 80% budget offers
  // 10 - (1 - 0.8) * 24 = 5.2, not 8.
  const busy = aggregateUsableFreeVramGb(
    [{ memoryFreeGb: 10, memoryTotalGb: 24, sharedMemory: false }],
    0.8,
  );
  assert.ok(Math.abs(busy - 5.2) < 0.01, `${busy}`);
});

test("an empty inventory is no verdict, not a fit", () => {
  assert.equal(aggregateUsableFreeVramGb([], 1), 0);
  assert.equal(classifyMemoryFit(8 * GB, aggregateUsableFreeVramGb([], 1)), "unknown");
});

// ---------------------------------------------------------------------------
// PART 3: readings that arrive broken

test("missing fields on a device read as nothing probed, and do not throw", () => {
  assert.doesNotThrow(() => aggregateUsableFreeVramGb([{}, {}], 1));
  assert.equal(aggregateUsableFreeVramGb([{}, {}], 1), 0);
  assert.equal(aggregateGpuMemoryTotalGb([{}, {}]), 0);
});

test("null, string, negative, NaN and Infinity readings never become capacity", () => {
  const bad = [null, "16", -8, Number.NaN, Number.POSITIVE_INFINITY, undefined];
  for (const value of bad) {
    const devices = [
      { memoryFreeGb: value as number, memoryTotalGb: value as number, sharedMemory: false },
      { memoryFreeGb: value as number, memoryTotalGb: value as number, sharedMemory: true },
    ];
    const free = aggregateUsableFreeVramGb(devices, 1);
    assert.ok(
      Number.isFinite(free) && free >= 0,
      `free from ${String(value)} was ${free}`,
    );
    assert.equal(free, 0, `free from ${String(value)}`);
    const total = aggregateGpuMemoryTotalGb([
      { memory_total_gb: value as number, shared_memory: false },
      { memory_total_gb: value as number, shared_memory: true },
    ]);
    assert.ok(Number.isFinite(total) && total >= 0, `total from ${String(value)}`);
    assert.equal(total, 0);
    // And no verdict is drawn from any of them.
    assert.equal(classifyMemoryFit(8 * GB, free), "unknown");
    assert.equal(classifyMemoryFit(8 * GB, total), "unknown");
  }
});

test("one broken device does not poison the readings beside it", () => {
  const free = aggregateUsableFreeVramGb(
    [DGPU, { memoryFreeGb: Number.NaN, memoryTotalGb: Number.NaN, sharedMemory: false }],
    1,
  );
  assert.ok(Math.abs(free - usableFreeVramGb(15, 16, 1)) < 0.01, `${free}`);
});

test("a non-finite budget fraction leaves the reading alone", () => {
  for (const fraction of [Number.NaN, Number.POSITIVE_INFINITY, 0, -1, 2]) {
    const free = aggregateUsableFreeVramGb([DGPU], fraction);
    assert.ok(Number.isFinite(free) && free >= 0, `fraction=${fraction} -> ${free}`);
    assert.ok(Math.abs(free - usableFreeVramGb(15, 16, 1)) < 0.01);
  }
});

// ---------------------------------------------------------------------------
// PART 3: the capacity resolver on the same shapes

const HOST = {
  hostGpuTotalGb: 27,
  hostSharesSystemRam: true,
  systemRamTotalGb: 96,
  unifiedMemory: false,
};

test("Apple's 64 GB pool is not discounted twice by a stale server budget", () => {
  // Apple reports the one pool as the GPU budget already, so the VRAM Budget slider
  // caps it once. The total must follow that capped figure and must NOT then prefer a
  // separately reported RAM number, which would hand back the very memory the slider
  // just took away.
  const capacity = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 64,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: true,
    gpuBudgetFraction: 0.5,
  });
  assert.equal(capacity.gpuCapacityGb, 32);
  assert.equal(capacity.totalCapacityGb, 32);
  assert.equal(capacity.singleMemoryPool, true);
});

test("one Vulkan iGPU with a 12 GB allowance on a 64 GB system", () => {
  // The allowance is a capped view of the RAM, so the RAM is not added on top -- and
  // the machine is not shrunk to the allowance either. The pool is the RAM.
  const capacity = resolveMemoryCapacityGb({
    pinnedDevices: [{ memoryTotalGb: 12, sharedMemory: true }],
    hostGpuTotalGb: 12,
    hostSharesSystemRam: true,
    systemRamTotalGb: 64,
    unifiedMemory: false,
  });
  assert.equal(capacity.gpuCapacityGb, 12);
  assert.equal(capacity.totalCapacityGb, 64);
  assert.equal(capacity.singleMemoryPool, true);
});

test("16 GB discrete plus a shared iGPU: each pin, and no pin", () => {
  const discretePin = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [{ memoryTotalGb: 16, sharedMemory: false }],
  });
  // Nothing is sharing that RAM with the pinned card, so RAM is a pool beside it.
  assert.equal(discretePin.gpuCapacityGb, 16);
  assert.equal(discretePin.totalCapacityGb, 112);
  assert.equal(discretePin.singleMemoryPool, false);

  const igpuPin = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [{ memoryTotalGb: 11, sharedMemory: true }],
  });
  assert.equal(igpuPin.gpuCapacityGb, 11);
  assert.equal(igpuPin.totalCapacityGb, 96);
  assert.equal(igpuPin.singleMemoryPool, true);

  const unpinned = resolveMemoryCapacityGb({ ...HOST, pinnedDevices: [] });
  // devices.some(...) makes the host read as shared, so RAM is not added on top.
  assert.equal(unpinned.gpuCapacityGb, 27);
  assert.equal(unpinned.totalCapacityGb, 96);
  assert.equal(unpinned.singleMemoryPool, true);
});

test("a pin whose devices all report 0 falls back to the host aggregate", () => {
  const unsized = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [
      { memoryTotalGb: 0, sharedMemory: false },
      { memoryTotalGb: 0, sharedMemory: false },
    ],
  });
  assert.equal(unsized.gpuCapacityGb, HOST.hostGpuTotalGb);
  // And the pool question falls back with it, rather than being answered by devices
  // that reported nothing.
  assert.equal(unsized.singleMemoryPool, true);
});

test("a capacity resolver handed nothing gives no verdict rather than a fit", () => {
  const nothing = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 0,
    hostSharesSystemRam: false,
    systemRamTotalGb: 0,
    unifiedMemory: false,
  });
  assert.equal(classifyMemoryFit(8 * GB, nothing.gpuCapacityGb), "unknown");
  assert.equal(classifyMemoryFit(8 * GB, nothing.totalCapacityGb), "unknown");
});

test("non-finite capacity inputs never produce a fit", () => {
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
    const capacity = resolveMemoryCapacityGb({
      pinnedDevices: [{ memoryTotalGb: value, sharedMemory: false }],
      hostGpuTotalGb: value,
      hostSharesSystemRam: false,
      systemRamTotalGb: value,
      unifiedMemory: false,
    });
    assert.ok(
      Number.isFinite(capacity.gpuCapacityGb) && capacity.gpuCapacityGb >= 0,
      `gpu ${capacity.gpuCapacityGb} from ${value}`,
    );
    assert.ok(
      Number.isFinite(capacity.totalCapacityGb) && capacity.totalCapacityGb >= 0,
      `total ${capacity.totalCapacityGb} from ${value}`,
    );
    assert.notEqual(classifyMemoryFit(8 * GB, capacity.gpuCapacityGb), "fits");
    assert.notEqual(classifyMemoryFit(8 * GB, capacity.totalCapacityGb), "fits");
  }
});

// ---------------------------------------------------------------------------
// PART 3: a persisted pin that no longer matches the hardware
//
// selectedGpuIds is remembered per model and survives a driver change, a card being
// removed, and a switch between the torch and the Vulkan index namespaces. The page
// resolves it with `gpuDevices.filter((d) => pinnedIds.includes(d.index))`, so this
// is that filter against a malformed list.

const INVENTORY = [
  { index: 0, memoryFreeGb: 15, memoryTotalGb: 16, sharedMemory: false },
  { index: 1, memoryFreeGb: 11, memoryTotalGb: 11, sharedMemory: true },
];

const pinned = (ids: number[]) =>
  ids.length > 0 ? INVENTORY.filter((d) => ids.includes(d.index)) : INVENTORY;

test("a pin naming indices that no longer exist falls back to the host", () => {
  const devices = pinned([7, 9]);
  assert.equal(devices.length, 0);
  const capacity = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: devices.map((d) => ({
      memoryTotalGb: d.memoryTotalGb,
      sharedMemory: d.sharedMemory,
    })),
  });
  // pinGoverns is false, so the whole-host answer stands rather than a 0 that would
  // hide the verdict entirely.
  assert.equal(capacity.gpuCapacityGb, HOST.hostGpuTotalGb);
  assert.equal(aggregateUsableFreeVramGb(devices, 1), 0);
});

test("a duplicated index selects its device once, not twice", () => {
  const devices = pinned([0, 0, 0]);
  assert.equal(devices.length, 1);
  assert.equal(
    aggregateUsableFreeVramGb(devices, 1),
    aggregateUsableFreeVramGb(pinned([0]), 1),
  );
});

test("negative and non-integer indices match nothing and do not throw", () => {
  assert.doesNotThrow(() => pinned([-1, -2, 1.5]));
  assert.equal(pinned([-1, -2, 1.5]).length, 0);
});

test("a pin mixing one real index with junk keeps only the real one", () => {
  const devices = pinned([1, -3, 42]);
  assert.equal(devices.length, 1);
  const capacity = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: devices.map((d) => ({
      memoryTotalGb: d.memoryTotalGb,
      sharedMemory: d.sharedMemory,
    })),
  });
  // The iGPU alone, so one pool, and RAM is not offered a second time.
  assert.equal(capacity.gpuCapacityGb, 11);
  assert.equal(capacity.singleMemoryPool, true);
});
