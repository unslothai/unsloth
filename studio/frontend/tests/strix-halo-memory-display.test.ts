// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { vramCards } from "../src/components/monitor-cards.ts";
import {
  carveOutVramGb,
  gpuMemoryTotalsGb,
} from "../src/hooks/gpu-vram.ts";
import type { GpuDevice } from "../src/hooks/use-system.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

// Ryzen AI Max+ 395 (Radeon 8060S) on Windows ROCm, 96 GB VGM out of 128 GB, as /api/system reports it: the driver
// pool is the carve-out plus 12.1 GiB of WDDM shared memory, and the OS sees 31.8 GiB of RAM with 11.3 GiB free.
const strixHalo = {
  index: 0,
  index_kind: "physical",
  name: "AMD Radeon(TM) 8060S Graphics",
  memory_total_gb: 107.92,
  shared_memory: true,
  unified_memory: true,
  shared_memory_host_backed_gb: 12.1,
  vram_used_gb: 1.16,
  vram_free_gb: 106.76,
} as GpuDevice;
const ramAvailableGb = 11.3;

test("Strix Halo: VRAM is the 95.8 GiB carve-out, and the shared part is RAM it can borrow", () => {
  const totals = gpuMemoryTotalsGb([strixHalo]);
  assert.deepEqual(totals, { dedicated: 95.82, shared: 12.1, total: 107.92 });
  assert.deepEqual(carveOutVramGb([strixHalo], 1.16, ramAvailableGb), {
    totalGb: 95.82,
    freeGb: 94.66,
    // Capped by free RAM: the 12.1 GiB shared pool comes out of the RAM tile's 31.8 GiB.
    borrowableGb: 11.3,
  });
  // The device row reads the same carve-out.
  assert.equal(
    carveOutVramGb([strixHalo], 1.16, null)?.totalGb,
    95.82,
  );
});

test("Strix Halo: the floating monitor divides by the carve-out, not carve-out plus shared RAM", () => {
  const vram = vramCards([strixHalo], { devices: [strixHalo] });
  assert.equal(vram.totalGb, 95.82, `floating monitor reads 1.16 GiB / ${vram.totalGb} GiB`);
});

test("usage past the carve-out leaves less RAM to borrow, and free stops at zero", () => {
  assert.deepEqual(carveOutVramGb([strixHalo], 100, null), {
    totalGb: 95.82,
    freeGb: 0,
    borrowableGb: 7.92,
  });
  assert.deepEqual(carveOutVramGb([strixHalo], null, 50), {
    totalGb: 95.82,
    freeGb: null,
    borrowableGb: 12.1,
  });
});

test("no carve-out reading for discrete cards, unified pools, or a small UMA slice under a big shared pool", () => {
  const discrete = { memory_total_gb: 31.86, shared_memory: false } as GpuDevice;
  const unified = { memory_total_gb: 64, unified_memory: true } as GpuDevice;
  // #8942: 2 GiB carve-out beside a 34.2 GiB shared pool keeps the pool as its GPU memory.
  const umaSlice = {
    memory_total_gb: 36.2,
    shared_memory: true,
    shared_memory_host_backed_gb: 34.2,
  } as GpuDevice;
  for (const device of [discrete, unified, umaSlice]) {
    assert.equal(carveOutVramGb([device], 1, 10), null);
  }
  assert.equal(vramCards([umaSlice], { devices: [umaSlice] }).totalGb, 36.2);
  // Two discrete cards beside that slice push the host's dedicated total past its shared pool; still no carve-out.
  const mixed = [
    umaSlice,
    { memory_total_gb: 24, shared_memory: false },
    { memory_total_gb: 24, shared_memory: false },
  ] as GpuDevice[];
  const mixedTotals = gpuMemoryTotalsGb(mixed);
  assert.ok(mixedTotals.dedicated >= mixedTotals.shared);
  assert.equal(carveOutVramGb(mixed, 1, 10), null);
});

test("Strix Halo: the VRAM tile reads used / carve-out, with the borrowable RAM on its own line", () => {
  const tab = readSrc("features/settings/tabs/resources-tab.tsx");
  // Was "1.16 GiB / 95.8 GiB VRAM + 12.1 GiB shared", truncated in the tile.
  // Only a carve-out the shared pool outgrows (a UMA slice) keeps the "VRAM + shared" split.
  assert.match(
    tab,
    /const vramCapacityLabel =\s*!metrics\.vramCarveOut && metrics\.vramShared > 0 && !unifiedPool\s*\? t\("settings\.resources\.environment\.vramWithShared"/,
  );
  assert.match(tab, /metrics\.vramCarveOut\?\.totalGb \?\? metrics\.vramTotal/);
  assert.match(
    tab,
    /gpu\.alsoSystemRam", \{\s*value: formatGiB\(metrics\.vramCarveOut\.borrowableGb\)/,
  );
  assert.match(tab, /const total = carveOut\?\.totalGb \?\? device\.memory_total_gb \?\? null/);
  assert.equal(
    en.settings.resources.gpu.alsoSystemRam.replace("{value}", "11.3 GiB"),
    "Can also borrow up to 11.3 GiB of system RAM",
  );
});

test("the GGUF row names the llama.cpp build, not a second system backend", () => {
  for (const file of [
    "features/settings/tabs/resources-tab.tsx",
    "components/floating-monitor.tsx",
  ]) {
    const source = readSrc(file);
    assert.match(source, /gpu\.llamaCppBuild", \{\s*backend: separateInferenceGpu\.backend/, file);
    // A Strix Halo's Vulkan budget is mostly the carve-out, not shared system RAM.
    assert.doesNotMatch(source, /sharedEstimatedAvailable/, file);
  }
  assert.equal(
    en.settings.resources.gpu.llamaCppBuild.replace("{backend}", "VULKAN"),
    "llama.cpp VULKAN build",
  );
});
