// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  deviceSharedMemorySplitGb,
  gpuMemoryTotalsGb,
} from "../src/hooks/gpu-vram.ts";
import type { GpuDevice } from "../src/hooks/use-system.ts";
import { readSrc } from "./helpers/kit.ts";

// #8942: Windows ROCm, RX 9070 XT beside a Ryzen iGPU with a 2 GiB carve-out and a 34.2 GiB shared pool.
const radeon: GpuDevice = {
  index_kind: "physical",
  memory_total_gb: 15.9,
  shared_memory: false,
  vram_used_gb: 3.28,
};
const igpu: GpuDevice = {
  index_kind: "physical",
  memory_total_gb: 36.2,
  shared_memory: true,
  shared_memory_host_backed_gb: 34.2,
  vram_used_gb: 1.5,
};

test("an iGPU row splits its total into carve-out VRAM and shared memory", () => {
  assert.deepEqual(deviceSharedMemorySplitGb(igpu), {
    dedicated: 2,
    shared: 34.2,
    total: 36.2,
  });
});

test("a discrete card's row keeps its plain VRAM total", () => {
  assert.equal(deviceSharedMemorySplitGb(radeon), null);
});

test("the rows add up to the VRAM tile's split", () => {
  const tile = gpuMemoryTotalsGb([radeon, igpu]);
  const split = deviceSharedMemorySplitGb(igpu)!;
  assert.equal(tile.dedicated, radeon.memory_total_gb! + split.dedicated);
  assert.equal(tile.shared, split.shared);
});

test("a Linux ROCm APU (unified, not flagged shared) reads as shared, as the tile does", () => {
  const apu: GpuDevice = {
    index_kind: "physical",
    memory_total_gb: 64,
    shared_memory: false,
    unified_memory: true,
  };
  assert.deepEqual(deviceSharedMemorySplitGb(apu), {
    dedicated: 0,
    shared: 64,
    total: 64,
  });
});

test("unknown or absent totals give no split", () => {
  assert.equal(
    deviceSharedMemorySplitGb({ shared_memory: true, memory_total_gb: 0 }),
    null,
  );
  assert.equal(deviceSharedMemorySplitGb({ shared_memory: true }), null);
});

test("the System device row labels a shared total instead of calling it VRAM", () => {
  const resourcesTab = readSrc("features/settings/tabs/resources-tab.tsx");
  assert.match(
    resourcesTab,
    /const sharedSplit = deviceSharedMemorySplitGb\(device\)/,
  );
  assert.match(
    resourcesTab,
    /\) : sharedSplit \? \([\s\S]*?vram: formatGiB\(sharedSplit\.dedicated\)[\s\S]*?shared: formatGiB\(sharedSplit\.shared\)[\s\S]*?gpu\.sharedWithSystemRam/,
  );
});
