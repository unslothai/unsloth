// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { gpuMemoryDisplay } from "../src/hooks/gpu-memory-display.ts";
import {
  gpuMemoryTotalsGb,
  resolveGpuVramUsedGb,
} from "../src/hooks/gpu-vram.ts";
import type { GpuDevice, SystemGpuInfo } from "../src/hooks/use-system.ts";

const integrated: GpuDevice = {
  index_kind: "vulkan",
  shared_memory: true,
  memory_total_gb: 25.69,
  vram_free_gb: 25.69,
};
const discrete: GpuDevice = {
  index_kind: "vulkan",
  shared_memory: false,
  memory_total_gb: 16,
  vram_used_gb: 1,
  vram_free_gb: 15,
};
const gpu = (devices: GpuDevice[], aggregate?: number): SystemGpuInfo => ({
  available: true,
  devices,
  vram_used_gb_aggregate: aggregate,
});

test("Vulkan iGPU presents allocation headroom without manufacturing usage", () => {
  const display = gpuMemoryDisplay(gpu([integrated]));
  assert.equal(display.sharedOnly, true);
  assert.equal(display.sharedAvailableGb, 25.69);
  assert.deepEqual(display.usageDevices, []);
  assert.equal(resolveGpuVramUsedGb(display.usageGpu), null);
});

test("mixed inventory measures dedicated usage against dedicated capacity only", () => {
  const input = gpu([discrete, integrated], 8);
  const display = gpuMemoryDisplay(input);
  assert.equal(display.sharedOnly, false);
  assert.equal(display.sharedAvailableGb, 25.69);
  const total = gpuMemoryTotalsGb(display.usageDevices).total;
  const used = resolveGpuVramUsedGb(display.usageGpu);
  assert.equal(total, 16);
  assert.equal(used, 1);
  assert.equal((used! / total) * 100, 6.25);
  assert.equal(input.devices.length, 2);
  assert.equal(gpuMemoryTotalsGb(input.devices).total, 41.69);
});

test("shared views count once and zero is a known reading", () => {
  assert.equal(
    gpuMemoryDisplay(gpu([integrated, { ...integrated, vram_free_gb: 12 }]))
      .sharedAvailableGb,
    25.69,
  );
  assert.equal(
    gpuMemoryDisplay(gpu([{ ...integrated, vram_free_gb: 0 }]))
      .sharedAvailableGb,
    0,
  );
});

test("missing or invalid shared free readings never fall back to capacity", () => {
  for (const free of [undefined, null, NaN, Infinity, -1]) {
    // null is also possible on the wire despite the optional-number API type.
    const device = { ...integrated, vram_free_gb: free } as GpuDevice;
    assert.equal(gpuMemoryDisplay(gpu([device])).sharedAvailableGb, null);
    assert.equal(
      gpuMemoryDisplay(gpu([integrated, device])).sharedAvailableGb,
      null,
    );
  }
});

test("ROCm host aggregate survives and unattributable rows remain unknown", () => {
  const devices = [
    { index_kind: "torch", memory_total_gb: 16 },
    { index_kind: "torch", memory_total_gb: 24 },
  ];
  const display = gpuMemoryDisplay(gpu(devices, 3));
  assert.equal(display.sharedOnly, false);
  assert.equal(resolveGpuVramUsedGb(display.usageGpu), 3);
  assert.equal(resolveGpuVramUsedGb(gpu(devices)), null);
  assert.equal(display.sharedDevices.length, 0);
});

test("host aggregate cannot supply usage for a filtered mixed inventory", () => {
  const display = gpuMemoryDisplay(
    gpu([{ ...discrete, vram_used_gb: undefined }, integrated], 3),
  );
  assert.equal(resolveGpuVramUsedGb(display.usageGpu), null);
});

test("other backends retain their shared-pool usage reporting", () => {
  for (const index_kind of ["torch", "metal", undefined]) {
    const device = { ...integrated, index_kind, vram_used_gb: 2 };
    const display = gpuMemoryDisplay(gpu([device]));
    assert.equal(display.sharedDevices.length, 0);
    assert.equal(gpuMemoryTotalsGb(display.usageDevices).total, 25.69);
    assert.equal(resolveGpuVramUsedGb(display.usageGpu), 2);
  }
  assert.deepEqual(gpuMemoryDisplay(undefined).usageDevices, []);
});
