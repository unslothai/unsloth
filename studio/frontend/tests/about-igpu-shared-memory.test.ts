// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { hardwareGpuMemorySplitGb } from "../src/hooks/gpu-vram.ts";

// /api/system/hardware rows from 2x Radeon AI PRO R9700 + a Ryzen 9 9950X iGPU on Windows ROCm.
const r9700 = {
  vramTotalGb: 31.86,
  sharedMemory: false,
  unifiedMemory: false,
  sharedMemoryHostBackedGb: null,
};
const igpu = {
  vramTotalGb: 36.23,
  sharedMemory: true,
  unifiedMemory: true,
  sharedMemoryHostBackedGb: 34.26,
};

test("a Windows ROCm iGPU splits into its carve-out and the shared system RAM", () => {
  const split = hardwareGpuMemorySplitGb(igpu);
  assert.ok(split);
  assert.equal(Math.round(split.dedicated), 2);
  assert.equal(Math.round(split.shared), 34);
  assert.equal(split.total, 36.23);
});

test("a discrete card is not split", () => {
  assert.equal(hardwareGpuMemorySplitGb(r9700), null);
});

test("a row from an older backend without the memory flags is not split", () => {
  assert.equal(hardwareGpuMemorySplitGb({ vramTotalGb: 36.23 }), null);
});

test("an unknown total is not split", () => {
  assert.equal(hardwareGpuMemorySplitGb({ ...igpu, vramTotalGb: null }), null);
});

test("the About tab labels the split with the System tab's wording", () => {
  const about = readFileSync(
    new URL("../src/features/settings/tabs/about-tab.tsx", import.meta.url),
    "utf8",
  );
  assert.match(about, /hardwareGpuMemorySplitGb\(gpu\)/);
  assert.match(about, /settings\.resources\.environment\.vramWithShared/);
});

test("the hardware route forwards the memory flags the split needs", () => {
  const main = readFileSync(
    new URL("../../backend/main.py", import.meta.url),
    "utf8",
  );
  const route = main.slice(main.indexOf('@app.get("/api/system/hardware")'));
  for (const key of [
    "shared_memory",
    "unified_memory",
    "shared_memory_host_backed_gb",
  ]) {
    assert.match(
      route.slice(0, route.indexOf("return body")),
      new RegExp(`"${key}"`),
    );
  }
});
