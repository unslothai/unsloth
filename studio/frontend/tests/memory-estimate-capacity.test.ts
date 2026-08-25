// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the Load-Model memory row measures its verdict against. The load-bearing
// case is a mixed Vulkan host: a discrete card beside an iGPU whose reported budget
// is a capped view of system RAM. Whether RAM counts as a pool beside the GPU is a
// question about the devices the load can reach, not about the host, and pinning
// changes which devices those are.

import assert from "node:assert/strict";
import test from "node:test";

import { resolveMemoryCapacityGb } from "../src/hooks/gpu-vram.ts";

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

test("pinning both counts the shared pool once and adds no RAM on top", () => {
  // 96, not 96 + 16: the discrete card's own memory is not added beside a pool it is
  // not part of. That under-counts the dGPU, which refuses a load rather than
  // admitting one that cannot run, and is the direction to be wrong in.
  const capacity = resolveMemoryCapacityGb({
    ...HOST,
    pinnedDevices: [DGPU, IGPU, IGPU],
  });
  assert.equal(capacity.gpuCapacityGb, 28);
  assert.equal(capacity.totalCapacityGb, 96);
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
