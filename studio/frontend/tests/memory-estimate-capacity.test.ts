// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether RAM is a separate pool depends on the pinned devices, not the host.

import assert from "node:assert/strict";
import test from "node:test";

import { resolveMemoryCapacityGb, usableFreeVramGb } from "../src/hooks/gpu-vram.ts";

// Mirrors studio/backend/tests/test_system_vulkan_gpu_info.py: a 16 GB dGPU and an iGPU
// whose 91 GiB raw total is reported as a 12 GB capped budget.
const DGPU = { memoryTotalGb: 16, sharedMemory: false };
const IGPU = { memoryTotalGb: 12, sharedMemory: true };
const HOST = {
  hostGpuTotalGb: 28,
  hostSharesSystemRam: true,
  systemRamTotalGb: 96,
  unifiedMemory: false,
};

test("a pin on the discrete card keeps system RAM as a pool beside it", () => {
  // The unpinned iGPU shares nothing with the dGPU, so RAM is a separate pool.
  const capacity = resolveMemoryCapacityGb({ ...HOST, pinnedDevices: [DGPU] });
  assert.equal(capacity.gpuCapacityGb, 16);
  assert.equal(capacity.totalCapacityGb, 112);
  assert.equal(capacity.singleMemoryPool, false);
});

test("a pin on the iGPU does not offer its own RAM twice, nor throw the rest away", () => {
  // The iGPU budget is already RAM: neither add 96 GB on top nor cap at 12 GB.
  const capacity = resolveMemoryCapacityGb({ ...HOST, pinnedDevices: [IGPU] });
  assert.equal(capacity.gpuCapacityGb, 12);
  assert.equal(capacity.totalCapacityGb, 96);
  assert.equal(capacity.singleMemoryPool, true);
});

test("pinning both keeps the discrete card as a pool beside system RAM", () => {
  // Count shared bytes once: 16 dedicated + 96 RAM, not 28 + 96.
  const capacity = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [DGPU, IGPU, IGPU],
  });
  assert.equal(capacity.singleMemoryPool, false);
  assert.equal(capacity.gpuCapacityGb, 28);
  assert.equal(capacity.totalCapacityGb, 112);
});

test("with no pin the host answers, and it says shared", () => {
  const capacity = resolveMemoryCapacityGb({ ...HOST, pinnedDevices: [] });
  assert.equal(capacity.gpuCapacityGb, 28);
  assert.equal(capacity.totalCapacityGb, 96);
});

test("a shared pool smaller than the GPU budget keeps the GPU budget", () => {
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
  // Unified memory: the GPU budget is the pool, so do not prefer a separate RAM figure.
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
  // 0 capacity reads as unknown, so empty pins fall back to the host.
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
  // 0 means nothing probed to callers, so it must not zero the capacity.
  assert.equal(resolveMemoryCapacityGb(base).gpuCapacityGb, 24);
  assert.equal(resolveMemoryCapacityGb({ ...base, gpuBudgetFraction: 0 }).gpuCapacityGb, 24);
  assert.equal(resolveMemoryCapacityGb({ ...base, gpuBudgetFraction: 1.5 }).gpuCapacityGb, 24);
  assert.equal(resolveMemoryCapacityGb({ ...base, gpuBudgetFraction: -1 }).gpuCapacityGb, 24);
});

// The loader subtracts an absolute reserve, not a fraction of free memory.
test("the budget is an absolute reserve, not a multiplier, on a busy card", () => {
  // _select_gpus example: 10 - (1 - 0.8) * 24 = 5.2.
  assert.ok(Math.abs(usableFreeVramGb(10, 24, 0.8) - 5.2) < 1e-9);
});

test("an idle card is where the two rules agree", () => {
  assert.ok(Math.abs(usableFreeVramGb(24, 24, 0.8) - 19.2) < 1e-9);
});

test("the floor keeps the budget monotonic on a small card", () => {
  // Capped at the default reserve so raising the slider never gives less.
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
  const mixed = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 28,
    hostDedicatedGpuTotalGb: 16,
    hostSharesSystemRam: false,
    systemRamTotalGb: 91,
    unifiedMemory: false,
  });
  assert.equal(mixed.singleMemoryPool, false);
  assert.equal(mixed.gpuCapacityGb, 28);
  // The iGPU's 12 GB budget is already inside the 91.
  assert.equal(mixed.totalCapacityGb, 107);

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
  const discrete = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostGpuTotalGb: 24,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: false,
  });
  assert.equal(discrete.singleMemoryPool, false);
  assert.equal(discrete.totalCapacityGb, 88);

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
