// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { vramCards } from "../src/components/monitor-cards.ts";
import {
  gpuMemoryDisplay,
  sharedMemoryDetail,
} from "../src/hooks/gpu-memory-display.ts";
import {
  gpuMemoryTotalsGb,
  gpuVramUsedIsPerDevice,
  resolveGpuVramUsedGb,
} from "../src/hooks/gpu-vram.ts";
import type { GpuDevice, SystemGpuInfo } from "../src/hooks/use-system.ts";

// `/api/system` on Windows 11, 2x Radeon AI PRO R9700 plus the Ryzen 9 9950X's iGPU: the iGPU
// has no usage counter.
const r9700 = (index: number, used: number): GpuDevice => ({
  index,
  name: "AMD Radeon AI PRO R9700",
  memory_total_gb: 31.86,
  vram_used_gb: used,
  vram_free_gb: Math.round((31.86 - used) * 100) / 100,
});
const igpu: GpuDevice = {
  index: 2,
  name: "AMD Radeon(TM) Graphics",
  memory_total_gb: 36.23,
  shared_memory: true,
  unified_memory: true,
  shared_memory_host_backed_gb: 34.26,
};
const gpu: SystemGpuInfo = {
  available: true,
  backend: "rocm",
  devices: [r9700(0, 21.75), r9700(1, 1.97), igpu],
  vram_used_gb_aggregate: null,
};

test("an unreadable shared iGPU leaves the discrete cards' usage on the VRAM tile", () => {
  // The same steps resources-tab.tsx and floating-monitor.tsx take for the tile.
  const display = gpuMemoryDisplay(gpu);
  assert.deepEqual(display.sharedDevices, [igpu]);
  assert.equal(display.sharedOnly, false);
  // Used, free and capacity all cover the same two cards, so the iGPU's pool never reads as free.
  assert.deepEqual(gpuMemoryTotalsGb(display.usageDevices), {
    dedicated: 63.72,
    shared: 0,
    total: 63.72,
  });
  assert.equal(gpuVramUsedIsPerDevice(display.usageDevices), true);
  const used = resolveGpuVramUsedGb(display.usageGpu);
  assert.ok(used !== null, "VRAM used must not be Unknown");
  assert.ok(Math.abs(used - 23.72) < 1e-9);
  // No free reading on the iGPU, so the tile keeps the shared label and drops the estimate line.
  assert.equal(display.sharedAvailableGb, null);
  const estimate = (gb: number) => `Estimated available: ${gb}`;
  assert.equal(
    sharedMemoryDetail(display.sharedAvailableGb, "Shared with system RAM", estimate),
    "Shared with system RAM",
  );
  assert.equal(
    sharedMemoryDetail(12.5, "Shared with system RAM", estimate),
    "Shared with system RAM\nEstimated available: 12.5",
  );
});

test("the monitor cards agree with the tile on this host", () => {
  const display = gpuMemoryDisplay(gpu);
  const cards = vramCards(display.usageDevices, display.usageGpu);
  assert.equal(cards.cards.length, 2);
  assert.ok(cards.usedGb !== null && Math.abs(cards.usedGb - 23.72) < 1e-9);
});

test("a host aggregate keeps the iGPU in the VRAM figure", () => {
  const display = gpuMemoryDisplay({ ...gpu, vram_used_gb_aggregate: 25.1 });
  assert.deepEqual(display.sharedDevices, []);
  assert.equal(resolveGpuVramUsedGb(display.usageGpu), 25.1);
});

test("an unreadable discrete card or an iGPU alone stays Unknown", () => {
  const unreadCard = gpuMemoryDisplay({
    available: true,
    devices: [r9700(0, 21.75), { ...r9700(1, 0), vram_used_gb: undefined }, igpu],
  });
  assert.deepEqual(unreadCard.sharedDevices, []);
  assert.equal(resolveGpuVramUsedGb(unreadCard.usageGpu), null);
  const apuOnly = gpuMemoryDisplay({ available: true, devices: [igpu] });
  assert.deepEqual(apuOnly.sharedDevices, []);
  assert.equal(resolveGpuVramUsedGb(apuOnly.usageGpu), null);
});
