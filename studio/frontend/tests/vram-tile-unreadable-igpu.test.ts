// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { vramCards } from "../src/components/monitor-cards.ts";
import { gpuMemoryDisplay } from "../src/hooks/gpu-memory-display.ts";
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
  const totals = gpuMemoryTotalsGb(display.usageDevices);
  assert.deepEqual(totals, { dedicated: 65.69, shared: 34.26, total: 99.95 });
  const used = resolveGpuVramUsedGb(display.usageGpu);
  assert.ok(used !== null, "VRAM used must not be Unknown");
  assert.ok(Math.abs(used - 23.72) < 1e-9);
  // Free stays derived from used, so the iGPU is counted as nothing used, not as unknown.
  assert.equal(gpuVramUsedIsPerDevice(display.usageDevices), false);
});

test("the monitor cards agree with the tile on this host", () => {
  const display = gpuMemoryDisplay(gpu);
  const cards = vramCards(display.usageDevices, display.usageGpu);
  assert.equal(cards.cards.length, 2);
  assert.ok(cards.usedGb !== null && Math.abs(cards.usedGb - 23.72) < 1e-9);
});

test("a host aggregate still wins over the partial discrete sum", () => {
  assert.equal(
    resolveGpuVramUsedGb({ ...gpu, vram_used_gb_aggregate: 25.1 }),
    25.1,
  );
});

test("an unreadable discrete card or an iGPU alone stays Unknown", () => {
  assert.equal(
    resolveGpuVramUsedGb({
      devices: [r9700(0, 21.75), { ...r9700(1, 0), vram_used_gb: undefined }, igpu],
    }),
    null,
  );
  assert.equal(resolveGpuVramUsedGb({ devices: [igpu] }), null);
});
