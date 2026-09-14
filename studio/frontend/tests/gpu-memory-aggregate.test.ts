// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type MemoryTotalDevice,
  aggregateGpuMemoryTotalGb,
} from "../src/hooks/gpu-vram.ts";

// 183359 MiB rounded to 2dp by the backend, the per-device figure a B200 reports.
const B200_GIB = 179.06;

function dedicated(count: number, memory = B200_GIB): MemoryTotalDevice[] {
  return Array.from({ length: count }, () => ({ memory_total_gb: memory }));
}

test("the total of several identical GPUs carries no float residue", () => {
  // Summing three 179.06 values is 537.1800000000001 before rounding, and the
  // run preview card prints the total without rounding it again.
  const total = aggregateGpuMemoryTotalGb(dedicated(3));
  assert.equal(total, 537.18);
  assert.equal(String(total), "537.18");
});

test("no total keeps more than the 2dp the devices arrived with", () => {
  for (let count = 1; count <= 8; count++) {
    const total = aggregateGpuMemoryTotalGb(dedicated(count));
    assert.equal(total, Math.round(total * 100) / 100, `${count} GPUs`);
  }
});

test("a shared host-memory pool is still counted once", () => {
  const devices: MemoryTotalDevice[] = [
    ...dedicated(1, 8.5),
    { memory_total_gb: 15.7, shared_memory: true },
    { memory_total_gb: 15.7, shared_memory: true },
  ];
  assert.equal(aggregateGpuMemoryTotalGb(devices), 24.2);
});

test("devices without a reported total do not poison the sum", () => {
  const devices: MemoryTotalDevice[] = [...dedicated(2), {}];
  assert.equal(aggregateGpuMemoryTotalGb(devices), 358.12);
});

test("no devices means no VRAM", () => {
  assert.equal(aggregateGpuMemoryTotalGb([]), 0);
});
