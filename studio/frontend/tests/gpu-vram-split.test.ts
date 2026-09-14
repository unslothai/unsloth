// SPDX-License-Identifier: Apache-2.0
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";

import {
  aggregateGpuMemoryTotalGb,
  gpuMemoryTotalsGb,
  gpuSharedHostMemoryGb,
  systemRamAvailableOutsideSharedPoolGb,
} from "../src/hooks/gpu-vram.ts";

test("dedicated and shared split the aggregate (#9242)", () => {
  const devices = [
    { memory_total_gb: 15.92 },
    { memory_total_gb: 12.15, shared_memory: true },
  ];
  const { dedicated, shared } = gpuMemoryTotalsGb(devices);
  const aggregate = aggregateGpuMemoryTotalGb(devices);

  assert.equal(dedicated, 15.92);
  assert.equal(shared, 12.15);
  assert.ok(
    Math.abs(dedicated + shared - aggregate) < 1e-9,
    "dedicated + shared must equal the aggregate the usage math uses",
  );
});

test("a shared pool counts once even when several devices report it", () => {
  const devices = [
    { memory_total_gb: 8 },
    {
      memory_total_gb: 12.15,
      shared_memory: true,
      shared_memory_host_backed_gb: 10.15,
    },
    {
      memory_total_gb: 12.15,
      shared_memory: true,
      shared_memory_host_backed_gb: 10.15,
    }, // same host pool, separate reserved heaps
  ];
  assert.deepEqual(gpuMemoryTotalsGb(devices), {
    dedicated: 12,
    shared: 10.15,
    total: 22.15,
  });
});

test("host RAM outside a shared pool remains available for CPU offload", () => {
  const devices = [{ memory_total_gb: 12.15, shared_memory: true }];
  const hostBackedGb = gpuSharedHostMemoryGb(devices);
  assert.equal(hostBackedGb, 12.15);
  assert.equal(systemRamAvailableOutsideSharedPoolGb(40, hostBackedGb), 27.85);
  assert.equal(systemRamAvailableOutsideSharedPoolGb(8, hostBackedGb), 0);
  assert.equal(systemRamAvailableOutsideSharedPoolGb(40, 0), 40);
});

test("reserved framebuffer memory is not subtracted from host RAM twice", () => {
  const devices = [
    {
      memory_total_gb: 89.47,
      shared_memory: true,
      shared_memory_host_backed_gb: 57.47,
    },
  ];
  assert.deepEqual(gpuMemoryTotalsGb(devices), {
    dedicated: 32,
    shared: 57.47,
    total: 89.47,
  });
  const hostBackedGb = gpuSharedHostMemoryGb(devices);
  assert.equal(hostBackedGb, 57.47);
  assert.equal(systemRamAvailableOutsideSharedPoolGb(64, hostBackedGb), 6.53);
});

test("fully host-backed unified memory is not counted twice", () => {
  const devices = [
    {
      memory_total_gb: 64,
      shared_memory: true,
      shared_memory_host_backed_gb: 64,
    },
  ];
  assert.deepEqual(gpuMemoryTotalsGb(devices), {
    dedicated: 0,
    shared: 64,
    total: 64,
  });
  assert.equal(
    systemRamAvailableOutsideSharedPoolGb(
      40,
      gpuSharedHostMemoryGb(devices),
    ),
    0,
  );
});

test("all-dedicated systems report zero shared", () => {
  const devices = [{ memory_total_gb: 24 }, { memory_total_gb: 24 }];
  assert.deepEqual(gpuMemoryTotalsGb(devices), {
    dedicated: 48,
    shared: 0,
    total: 48,
  });
});

test("float error does not leak into the halves", () => {
  // 2dp inputs; the derived halves must stay at the same precision.
  const devices = [{ memory_total_gb: 179.06 }, { memory_total_gb: 179.06 }];
  assert.equal(gpuMemoryTotalsGb(devices).dedicated, 358.12);
});
