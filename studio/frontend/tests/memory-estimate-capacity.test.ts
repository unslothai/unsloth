// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the Load-Model memory row measures its verdict against. The load-bearing
// case is a mixed Vulkan host: a discrete card beside an iGPU whose reported budget
// is a capped view of system RAM. Whether RAM counts as a pool beside the GPU is a
// question about the devices the load can reach, not about the host, and pinning
// changes which devices those are.

import assert from "node:assert/strict";
import test from "node:test";

import { resolveMemoryCapacityGb, usableFreeVramGb } from "../src/hooks/gpu-vram.ts";

// The inventory in studio/backend/tests/test_system_vulkan_gpu_info.py: an RX 9070 XT
// with its own 16 GB, and an 8060S iGPU whose 91 GiB raw total is reported as the
// 12 GB capped budget.
const DGPU = { memoryTotalGb: 16, sharedMemory: false };
const IGPU = { memoryTotalGb: 12, sharedMemory: true };
const HOST = {
  // What toGpuInfo aggregates for that pair: dedicated summed, shared counted once.
  hostGpuTotalGb: 28,
  // devices.some(...), so the whole host reads as shared while only one card is.
  hostSharesSystemRam: true,
  systemRamTotalGb: 96,
  unifiedMemory: false,
};

test("a pin on the discrete card keeps system RAM as a pool beside it", () => {
  // The iGPU is not in the load, so nothing is sharing that RAM with it and layers
  // that do not fit on the 16 GB card spill into the 96 GB host pool. Reading the
  // host-level shared flag here capped the total at 16 GB and called an 18 GB load
  // impossible with 96 GB of RAM standing free.
  const capacity = resolveMemoryCapacityGb({ ...HOST, pinnedDevices: [DGPU] });
  assert.equal(capacity.gpuCapacityGb, 16);
  assert.equal(capacity.totalCapacityGb, 112);
  assert.equal(capacity.singleMemoryPool, false);
});

test("a pin on the iGPU does not offer its own RAM twice, nor throw the rest away", () => {
  // Its 12 GB IS system RAM, already capped, so the 96 GB must not be ADDED on top --
  // that counts the same bytes twice and calls an oversized load a fit.
  //
  // It must not REPLACE it either, which is what this asserted before: the machine
  // genuinely has 96 GB, and capping its total at the iGPU's 12 GB allowance called a
  // 20 GB CPU-offloaded load impossible on a host with 76 GB to spare.
  const capacity = resolveMemoryCapacityGb({ ...HOST, pinnedDevices: [IGPU] });
  assert.equal(capacity.gpuCapacityGb, 12);
  assert.equal(capacity.totalCapacityGb, 96);
  assert.equal(capacity.singleMemoryPool, true);
});

test("pinning both keeps the discrete card as a pool beside system RAM", () => {
  // This asserted 96, on the reasoning that under-counting the dGPU refuses a load
  // rather than admitting one. That looked at half the effect: the flag it also sets
  // makes the row show a lone Shared figure and drop the GPU verdict, so a fixed
  // placement larger than the discrete card had nothing to catch it. Two pools now,
  // counting the shared bytes once: 16 dedicated + 96 RAM, not 28 + 96.
  const capacity = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [DGPU, IGPU, IGPU],
  });
  assert.equal(capacity.singleMemoryPool, false);
  assert.equal(capacity.gpuCapacityGb, 28);
  assert.equal(capacity.totalCapacityGb, 112);
});

test("with no pin the host answers, and it says shared", () => {
  // Same correction as the iGPU pin: shared means RAM is not added on top, not that
  // the machine shrinks to the iGPU's allowance.
  const capacity = resolveMemoryCapacityGb({ ...HOST, pinnedDevices: [] });
  assert.equal(capacity.gpuCapacityGb, 28);
  assert.equal(capacity.totalCapacityGb, 96);
});

test("a shared pool smaller than the GPU budget keeps the GPU budget", () => {
  // The guard against the correction going the other way: whatever the RAM figure is,
  // the pool is never reported as less than what the GPU is already allowed.
  const capacity = resolveMemoryCapacityGb({
    hostGpuTotalGb: 32,
    hostSharesSystemRam: true,
    systemRamTotalGb: 8,
    unifiedMemory: false,
    pinnedDevices: [],
  });
  assert.equal(capacity.totalCapacityGb, 32);
});

test("Apple's unified pool is still reported as the GPU budget alone", () => {
  // Unified memory is not the capped-view case: the GPU budget already IS the pool,
  // so the max() above must not start preferring a separately reported RAM figure.
  const capacity = resolveMemoryCapacityGb({
    hostGpuTotalGb: 36,
    hostSharesSystemRam: false,
    systemRamTotalGb: 36,
    unifiedMemory: true,
    pinnedDevices: [],
  });
  assert.equal(capacity.totalCapacityGb, 36);
  assert.equal(capacity.singleMemoryPool, true);
});

test("a plain multi-GPU host adds its RAM whether or not a card is pinned", () => {
  const plain = {
    hostGpuTotalGb: 48,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: false,
  };
  assert.equal(
    resolveMemoryCapacityGb({ ...plain, pinnedDevices: [] }).totalCapacityGb,
    112,
  );
  const pinned = resolveMemoryCapacityGb({
    ...plain,
    pinnedDevices: [{ memoryTotalGb: 24, sharedMemory: false }],
  });
  // The pin is the pool a split spreads over, and RAM is still beside it.
  assert.equal(pinned.gpuCapacityGb, 24);
  assert.equal(pinned.totalCapacityGb, 88);
});

test("Apple Silicon is one pool however the devices are described", () => {
  const capacity = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 128,
    hostSharesSystemRam: false,
    systemRamTotalGb: 128,
    unifiedMemory: true,
  });
  assert.equal(capacity.totalCapacityGb, 128);
  assert.equal(capacity.singleMemoryPool, true);
});

test("an unprobed host gives no verdict rather than a fit", () => {
  // 0 capacity is what classifyMemoryFit reads as "unknown". A pin whose devices
  // report nothing must fall back to the host rather than answer 0 on its own.
  const nothing = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 0,
    hostSharesSystemRam: false,
    systemRamTotalGb: 0,
    unifiedMemory: false,
  });
  assert.equal(nothing.gpuCapacityGb, 0);
  assert.equal(nothing.totalCapacityGb, 0);
  const unsized = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [{ memoryTotalGb: 0, sharedMemory: false }],
  });
  assert.equal(unsized.gpuCapacityGb, HOST.hostGpuTotalGb);
});

// The VRAM Budget slider sits in the same panel and caps what the next load may claim
// per GPU. Measuring the verdict against the raw total contradicted the control
// directly above the row: at 80% a 20 GiB footprint on a 24 GiB card is over the line
// the slider draws, and the row called it comfortable.
test("the configured budget caps the GPU capacity a verdict is measured against", () => {
  const capacity = resolveMemoryCapacityGb({
    hostGpuTotalGb: 24,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: false,
    pinnedDevices: [],
    gpuBudgetFraction: 0.8,
  });
  assert.equal(capacity.gpuCapacityGb, 19.2);
  // Host RAM is not the GPU's allowance, so the total follows the capped GPU figure
  // plus the whole of RAM.
  assert.equal(capacity.totalCapacityGb, 83.2);
});

test("the budget applies to a pinned subset too", () => {
  const capacity = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [DGPU],
    gpuBudgetFraction: 0.5,
  });
  assert.equal(capacity.gpuCapacityGb, 8);
});

test("an absent or nonsensical budget leaves the capacity alone", () => {
  const base = {
    hostGpuTotalGb: 24,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: false,
    pinnedDevices: [],
  };
  // A 0 or a missing value must not zero the capacity: every caller reads 0 as
  // "nothing probed" and would stop showing a verdict at all.
  assert.equal(resolveMemoryCapacityGb(base).gpuCapacityGb, 24);
  assert.equal(resolveMemoryCapacityGb({ ...base, gpuBudgetFraction: 0 }).gpuCapacityGb, 24);
  assert.equal(resolveMemoryCapacityGb({ ...base, gpuBudgetFraction: 1.5 }).gpuCapacityGb, 24);
  assert.equal(resolveMemoryCapacityGb({ ...base, gpuBudgetFraction: -1 }).gpuCapacityGb, 24);
});

// The loader subtracts an ABSOLUTE reserve from what is free; it does not scale free
// memory by the fraction. The two agree only on an idle card, and the gap is what
// decides whether a busy-card warning appears at all.
test("the budget is an absolute reserve, not a multiplier, on a busy card", () => {
  // _select_gpus' own example: 24 GB card, 10 GB free, 80% budget.
  // free - (1 - 0.8) * 24 = 10 - 4.8 = 5.2, where a multiplication says 8.
  assert.ok(Math.abs(usableFreeVramGb(10, 24, 0.8) - 5.2) < 1e-9);
});

test("an idle card is where the two rules agree", () => {
  assert.ok(Math.abs(usableFreeVramGb(24, 24, 0.8) - 19.2) < 1e-9);
});

test("the floor keeps the budget monotonic on a small card", () => {
  // Capped at the default's own reserve, so nudging the slider up never hands back
  // less. A flat 512 MiB would do exactly that on any card under about 17 GB.
  const atDefault = usableFreeVramGb(8, 8, 0.97);
  const justAbove = usableFreeVramGb(8, 8, 0.971);
  assert.ok(justAbove >= atDefault);
});

test("a probe with no total falls back to the fraction, as the loader does", () => {
  assert.ok(Math.abs(usableFreeVramGb(10, 0, 0.8) - 8) < 1e-9);
});

test("the usable figure never goes negative", () => {
  assert.equal(usableFreeVramGb(0.1, 24, 0.5), 0);
});

test("a mixed dedicated and shared inventory is two pools, not one", () => {
  // devices.some(shared) marked a discrete card sitting beside a Vulkan iGPU as one
  // pool. The row shows a lone Shared figure there and drops the GPU verdict
  // entirely, so a fixed Manual placement larger than the discrete card read as a fit
  // against a ceiling it can never spill into.
  const mixed = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 28, // 16 GB card + a 12 GB iGPU budget, counted once
    hostDedicatedGpuTotalGb: 16,
    hostSharesSystemRam: false, // every(), so a mixed host is no longer flagged
    systemRamTotalGb: 91,
    unifiedMemory: false,
  });
  assert.equal(mixed.singleMemoryPool, false);
  assert.equal(mixed.gpuCapacityGb, 28);
  // 16 + 91, NOT 28 + 91: the iGPU's 12 GB budget is already inside the 91.
  assert.equal(mixed.totalCapacityGb, 107);

  // An all-shared host is still one pool, and still takes the max rather than a sum.
  const allShared = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 12,
    hostDedicatedGpuTotalGb: 0,
    hostSharesSystemRam: true,
    systemRamTotalGb: 91,
    unifiedMemory: false,
  });
  assert.equal(allShared.singleMemoryPool, true);
  assert.equal(allShared.totalCapacityGb, 91);
});

test("a pin decides the pool question for the cards it names", () => {
  const dedicated = { memoryTotalGb: 16, sharedMemory: false };
  const igpu = { memoryTotalGb: 12, sharedMemory: true };

  // Only the discrete card: two pools, and RAM is genuinely beside it.
  const onCard = resolveMemoryCapacityGb({
    pinnedDevices: [dedicated],
    hostGpuTotalGb: 28,
    hostDedicatedGpuTotalGb: 16,
    hostSharesSystemRam: false,
    systemRamTotalGb: 91,
    unifiedMemory: false,
  });
  assert.equal(onCard.singleMemoryPool, false);
  assert.equal(onCard.totalCapacityGb, 107);

  // Only the iGPU: one pool.
  const onIgpu = resolveMemoryCapacityGb({
    pinnedDevices: [igpu],
    hostGpuTotalGb: 28,
    hostDedicatedGpuTotalGb: 16,
    hostSharesSystemRam: false,
    systemRamTotalGb: 91,
    unifiedMemory: false,
  });
  assert.equal(onIgpu.singleMemoryPool, true);
  assert.equal(onIgpu.totalCapacityGb, 91);

  // Both pinned: still two pools, because the discrete VRAM is still there.
  const onBoth = resolveMemoryCapacityGb({
    pinnedDevices: [dedicated, igpu],
    hostGpuTotalGb: 28,
    hostDedicatedGpuTotalGb: 16,
    hostSharesSystemRam: false,
    systemRamTotalGb: 91,
    unifiedMemory: false,
  });
  assert.equal(onBoth.singleMemoryPool, false);
  assert.equal(onBoth.gpuCapacityGb, 28);
  assert.equal(onBoth.totalCapacityGb, 107);
});

test("a partially host-backed APU keeps its reserved memory beside system RAM", () => {
  const apu = {
    memoryTotalGb: 100,
    sharedMemory: true,
    sharedMemoryHostBackedGb: 92,
  };
  const capacity = resolveMemoryCapacityGb({
    pinnedDevices: [apu],
    hostGpuTotalGb: 100,
    hostDedicatedGpuTotalGb: 8,
    hostSharesSystemRam: false,
    systemRamTotalGb: 128,
    unifiedMemory: false,
  });
  assert.equal(capacity.gpuCapacityGb, 100);
  assert.equal(capacity.totalCapacityGb, 136);
  assert.equal(capacity.singleMemoryPool, false);
});

test("the gpu budget does not discount a partial APU reserve twice", () => {
  const apu = {
    memoryTotalGb: 100,
    sharedMemory: true,
    sharedMemoryHostBackedGb: 92,
  };
  const common = {
    hostDevices: [apu],
    hostGpuTotalGb: 100,
    hostDedicatedGpuTotalGb: 8,
    hostSharesSystemRam: false,
    systemRamTotalGb: 128,
    unifiedMemory: false,
    gpuBudgetFraction: 0.8,
  };
  const pinned = resolveMemoryCapacityGb({
    ...common,
    pinnedDevices: [apu],
  });
  const unpinned = resolveMemoryCapacityGb({
    ...common,
    pinnedDevices: [],
  });
  assert.equal(pinned.gpuCapacityGb, 80);
  assert.equal(pinned.totalCapacityGb, 136);
  assert.deepEqual(unpinned, pinned);
});

test("the gpu budget is applied to each APU sharing one host pool", () => {
  const apu = {
    memoryTotalGb: 100,
    sharedMemory: true,
    sharedMemoryHostBackedGb: 92,
  };
  const common = {
    hostDevices: [apu, apu],
    hostGpuTotalGb: 108,
    hostDedicatedGpuTotalGb: 16,
    hostSharesSystemRam: false,
    systemRamTotalGb: 128,
    unifiedMemory: false,
    gpuBudgetFraction: 0.8,
  };
  const pinned = resolveMemoryCapacityGb({
    ...common,
    pinnedDevices: [apu, apu],
  });
  const unpinned = resolveMemoryCapacityGb({
    ...common,
    pinnedDevices: [],
  });
  assert.equal(pinned.gpuCapacityGb, 108);
  assert.equal(pinned.totalCapacityGb, 144);
  assert.deepEqual(unpinned, pinned);
});

test("duplicate fully shared Vulkan views do not erase the gpu budget", () => {
  const igpu = {
    memoryTotalGb: 12,
    sharedMemory: true,
  };
  const capacity = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostDevices: [igpu, igpu],
    hostGpuTotalGb: 12,
    hostDedicatedGpuTotalGb: 0,
    hostSharesSystemRam: true,
    systemRamTotalGb: 12,
    unifiedMemory: false,
    gpuBudgetFraction: 0.8,
  });
  assert.equal(capacity.gpuCapacityGb, 9.6);
  assert.equal(capacity.totalCapacityGb, 12);
  assert.equal(capacity.singleMemoryPool, true);
});

test("a discrete-only host is unchanged by the dedicated-only ceiling", () => {
  // The regression guard: with no shared device the two figures are equal, so every
  // ordinary machine keeps exactly the ceiling it had.
  const discrete = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 24,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: false,
  });
  assert.equal(discrete.singleMemoryPool, false);
  assert.equal(discrete.totalCapacityGb, 88);

  // Apple keeps its own branch, whatever the devices say.
  const apple = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 64,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: true,
  });
  assert.equal(apple.singleMemoryPool, true);
  assert.equal(apple.totalCapacityGb, 64);
});
