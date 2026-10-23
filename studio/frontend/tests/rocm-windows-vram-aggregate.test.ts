// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  gpuVramUsedIsPerDevice,
  resolveGpuVramUsedGb,
  type VramReportingDevice,
  type VramReportingGpu,
} from "../src/hooks/gpu-vram.ts";

interface SystemGpuPayload extends VramReportingGpu {
  available: boolean;
  backend?: string;
  devices?: (VramReportingDevice & {
    index?: number;
    name?: string;
    memory_total_gb?: number;
  })[];
}

// Windows ROCm host: LUID VRAM counters are not keyed to torch ordinals, so per-device is Unknown.
function reporterGpu(
  overrides: Partial<SystemGpuPayload> = {},
): SystemGpuPayload {
  return {
    available: true,
    backend: "rocm",
    devices: [
      { index: 0, name: "AMD Radeon PRO W7900", memory_total_gb: 45.0 },
      { index: 1, name: "AMD Radeon PRO W7500", memory_total_gb: 7.98 },
    ],
    vram_used_gb_aggregate: 0.36,
    ...overrides,
  };
}

test("unattributable per-device usage still reports the host total", () => {
  assert.equal(resolveGpuVramUsedGb(reporterGpu()), 0.36);
  assert.equal(gpuVramUsedIsPerDevice(reporterGpu().devices ?? []), false);
});

test("per-device usage wins over the aggregate when every device reports", () => {
  const gpu: SystemGpuPayload = reporterGpu({
    devices: [
      {
        index: 0,
        memory_total_gb: 45.0,
        vram_used_gb: 40.0,
      },
      { index: 1, memory_total_gb: 7.98, vram_used_gb: 0.5 },
    ],
    // Not the per-device sum, so the test pins which source wins.
    vram_used_gb_aggregate: 99.0,
  });
  assert.equal(gpuVramUsedIsPerDevice(gpu.devices ?? []), true);
  assert.equal(resolveGpuVramUsedGb(gpu), 40.5);
});

test("a partially attributed pair falls back to the aggregate, not a short sum", () => {
  const gpu: SystemGpuPayload = reporterGpu({
    devices: [
      { index: 0, memory_total_gb: 45.0, vram_used_gb: 40.0 },
      { index: 1, memory_total_gb: 7.98 },
    ],
    vram_used_gb_aggregate: 40.5,
  });
  assert.equal(resolveGpuVramUsedGb(gpu), 40.5);
});

test("no aggregate stays unknown rather than becoming zero", () => {
  assert.equal(
    resolveGpuVramUsedGb(reporterGpu({ vram_used_gb_aggregate: null })),
    null,
  );
  assert.equal(
    resolveGpuVramUsedGb(reporterGpu({ vram_used_gb_aggregate: undefined })),
    null,
  );
  assert.equal(resolveGpuVramUsedGb(null), null);
  assert.equal(
    resolveGpuVramUsedGb({ devices: [] }),
    null,
  );
});
