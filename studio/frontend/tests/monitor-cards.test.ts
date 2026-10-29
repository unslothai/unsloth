// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { shortCardName, vramCards } from "../src/components/monitor-cards.ts";
import type { GpuDevice } from "../src/hooks/use-system.ts";
import { readSrc } from "./helpers/kit.ts";

// GR33N as /api/system reported it (Windows ROCm): two R9700s and the iGPU, which has no usage reading.
const r9700 = (index: number, used: number): GpuDevice =>
  ({ index, index_kind: "physical", name: "AMD Radeon AI PRO R9700", memory_total_gb: 31.86,
     shared_memory: false, unified_memory: false, vram_used_gb: used, vram_free_gb: 31.86 - used }) as GpuDevice;
const igpu = { index: 2, index_kind: "physical", name: "AMD Radeon(TM) Graphics", memory_total_gb: 36.23,
  shared_memory: true, unified_memory: true, shared_memory_host_backed_gb: 34.26 } as GpuDevice;

test("an iGPU beside dedicated cards neither blanks nor pads the VRAM total", () => {
  const devices = [r9700(0, 6.62), r9700(1, 7.73), igpu];
  const vram = vramCards(devices, { devices });
  assert.equal(vram.cards.length, 2);
  assert.equal(vram.totalGb, 63.72);
  assert.ok(Math.abs((vram.usedGb ?? 0) - 14.35) < 1e-9);
  assert.deepEqual(vram.rows.map((row) => [row.name, row.usedGb, row.totalGb]), [
    ["0 · AI PRO R9700", 6.62, 31.86],
    ["1 · AI PRO R9700", 7.73, 31.86],
  ]);
});

test("one card gets no per-card rows, and a card without a reading stays unknown", () => {
  const one = [r9700(0, 4)];
  assert.deepEqual(vramCards(one, { devices: one }).rows, []);
  const blind = [{ ...r9700(0, 0), vram_used_gb: undefined }, r9700(1, 2)] as GpuDevice[];
  const vram = vramCards(blind, { devices: blind });
  assert.equal(vram.usedGb, null);
  assert.equal(vram.rows[0].usedGb, null);
});

test("shared-only hosts keep their devices: there is nothing dedicated to prefer", () => {
  assert.equal(vramCards([igpu], { devices: [igpu] }).cards.length, 1);
});

test("the floating monitor and Settings both read VRAM through vramCards", () => {
  assert.match(readSrc("components/floating-monitor.tsx"), /vramCards\(devices, memoryDisplay\.usageGpu\)/);
  assert.match(readSrc("features/settings/tabs/resources-tab.tsx"), /vramCards\(memoryDisplay\.usageDevices, memoryDisplay\.usageGpu\)/);
  assert.match(readSrc("components/floating-monitor.tsx"), /<CardRows rows=\{vram\.rows\} \/>/);
});

test("card names drop the vendor prefix every row would repeat", () => {
  assert.equal(shortCardName("NVIDIA GeForce RTX 4090"), "RTX 4090");
  assert.equal(shortCardName("NVIDIA A100-SXM4-80GB"), "A100-SXM4-80GB");
  assert.equal(shortCardName("Intel(R) Arc(TM) A770 Graphics"), "A770 Graphics");
  assert.equal(shortCardName("Apple M5"), "Apple M5");
  assert.equal(shortCardName(undefined), "GPU");
});
