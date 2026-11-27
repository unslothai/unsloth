// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { hardwareGpuMemoryGb } from "../src/hooks/gpu-vram.ts";

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

test("a Windows ROCm iGPU shows its own carve-out, not the system RAM it can borrow", () => {
  const memory = hardwareGpuMemoryGb(igpu);
  assert.ok(memory);
  assert.equal(Math.round(memory.gb), 2);
  assert.equal(memory.integrated, true);
});

test("a discrete card shows its full VRAM and is not marked integrated", () => {
  assert.deepEqual(hardwareGpuMemoryGb(r9700), {
    gb: 31.86,
    integrated: false,
  });
});

test("an integrated GPU with no carve-out of its own keeps its whole total", () => {
  const gb10 = {
    vramTotalGb: 119.7,
    sharedMemory: true,
    unifiedMemory: true,
    sharedMemoryHostBackedGb: 119.7,
  };
  assert.deepEqual(hardwareGpuMemoryGb(gb10), { gb: 119.7, integrated: true });
});

test("a row from an older backend without the memory flags keeps the plain total", () => {
  assert.deepEqual(hardwareGpuMemoryGb({ vramTotalGb: 36.23 }), {
    gb: 36.23,
    integrated: false,
  });
});

test("an unknown total shows nothing", () => {
  assert.equal(hardwareGpuMemoryGb({ ...igpu, vramTotalGb: null }), null);
});

test("the hardware route forwards the memory flags", () => {
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
