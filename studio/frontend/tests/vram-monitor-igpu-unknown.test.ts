// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { gpuMemoryDisplay } from "../src/hooks/gpu-memory-display.ts";
import {
  gpuMemoryTotalsGb,
  resolveGpuVramUsage,
  resolveGpuVramUsedGb,
} from "../src/hooks/gpu-vram.ts";
import type { GpuDevice, SystemGpuInfo } from "../src/hooks/use-system.ts";

// Windows 11 ROCm: 2x Radeon AI PRO R9700 plus the Ryzen 9 9950X iGPU (gfx1036). The
// iGPU is 1.97 GiB carved out plus 34.3 GiB of shared host RAM and reports no usage,
// and there is no host aggregate, so the Live monitor tile read "Unknown / 65.7 GiB
// VRAM + 34.3 GiB shared" while both cards had real readings.
const r9700 = (index: number, used: number | undefined): GpuDevice => ({
  index,
  index_kind: "relative",
  name: "AMD Radeon AI PRO R9700",
  memory_total_gb: 31.9,
  vram_used_gb: used,
});
const igpu: GpuDevice = {
  index: 2,
  index_kind: "relative",
  name: "AMD Radeon(TM) Graphics",
  memory_total_gb: 36.27,
  shared_memory: true,
  shared_memory_host_backed_gb: 34.3,
};
const gpu = (
  devices: GpuDevice[],
  aggregate: number | null = null,
): SystemGpuInfo => ({
  available: true,
  backend: "rocm",
  devices,
  vram_used_gb_aggregate: aggregate,
});

test("reporter's host: the cards' usage is shown against the cards' VRAM", () => {
  const input = gpu([r9700(0, 0), r9700(1, 2.15), igpu]);
  const display = gpuMemoryDisplay(input);
  // A ROCm iGPU is not a Vulkan headroom device, so it stays in the usage set.
  assert.equal(display.usageDevices.length, 3);
  assert.equal(resolveGpuVramUsedGb(display.usageGpu), null);

  const usage = resolveGpuVramUsage(display.usageGpu);
  assert.equal(usage.usedGb, 2.15);
  assert.deepEqual(
    usage.devices.map((device) => device.index),
    [0, 1],
  );
  const totals = gpuMemoryTotalsGb(usage.devices);
  assert.deepEqual(totals, { dedicated: 63.8, shared: 0, total: 63.8 });
  assert.equal(Math.round((usage.usedGb! / totals.total) * 100), 3);
  // The per-device list reads displayedGpu.devices, which still has the iGPU.
  assert.equal(input.devices.length, 3);
});

test("the full inventory total is unchanged when usage is unknown", () => {
  const usage = resolveGpuVramUsage(
    gpu([r9700(0, 0), r9700(1, undefined), igpu]),
  );
  assert.equal(usage.usedGb, null);
  assert.equal(usage.devices.length, 3);
  assert.deepEqual(gpuMemoryTotalsGb(usage.devices), {
    dedicated: 65.77,
    shared: 34.3,
    total: 100.07,
  });
});

test("an unreported discrete card is never filled in", () => {
  const usage = resolveGpuVramUsage(
    gpu([r9700(0, 2.15), r9700(1, undefined), igpu]),
  );
  assert.equal(usage.usedGb, null);
});

test("a Linux ROCm APU (unified_memory only) counts as shared too", () => {
  const apu: GpuDevice = {
    index: 1,
    name: "AMD Radeon 8060S",
    memory_total_gb: 64,
    unified_memory: true,
  };
  const usage = resolveGpuVramUsage(gpu([r9700(0, 4), apu]));
  assert.equal(usage.usedGb, 4);
  assert.equal(usage.devices.length, 1);
});

test("nothing reporting stays unknown rather than 0", () => {
  const usage = resolveGpuVramUsage(gpu([r9700(0, undefined), igpu]));
  assert.equal(usage.usedGb, null);
  assert.equal(resolveGpuVramUsage(gpu([igpu])).usedGb, null);
  assert.equal(resolveGpuVramUsage(null).usedGb, null);
});

test("a host aggregate still covers the whole inventory before any partial sum", () => {
  const usage = resolveGpuVramUsage(
    gpu([r9700(0, 0), r9700(1, 2.15), igpu], 3.4),
  );
  assert.equal(usage.usedGb, 3.4);
  assert.equal(usage.devices.length, 3);
});

test("every device reporting sums all of them, iGPU included", () => {
  const usage = resolveGpuVramUsage(
    gpu([r9700(0, 1), r9700(1, 2), { ...igpu, vram_used_gb: 0.5 }]),
  );
  assert.equal(usage.usedGb, 3.5);
  assert.equal(usage.devices.length, 3);
});
