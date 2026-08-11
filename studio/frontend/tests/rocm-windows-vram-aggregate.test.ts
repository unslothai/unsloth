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

// The payload shape as /api/system actually sends it: the helpers read two of
// these fields structurally and the rest ride along, which is what the tile gets.
interface SystemGpuPayload extends VramReportingGpu {
  available: boolean;
  backend?: string;
  devices?: (VramReportingDevice & {
    index?: number;
    name?: string;
    memory_total_gb?: number;
  })[];
}

// Issue #7452, reported on a Windows 10 ROCm host with an AMD Radeon PRO W7900
// (45 GiB) beside a W7500 (7.98 GiB). Windows shares no key between the LUID VRAM
// counters and torch ordinals, so a usage that fits both cards cannot be pinned to
// either and every device reads Unknown -- which on that pair is idle and every
// small model. The backend still knows the host total, and the tile rendered
// Unknown anyway.
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
    // Deliberately NOT 40.5. An aggregate equal to the per-device sum would pass
    // whichever source won, so it would not pin the precedence at all.
    vram_used_gb_aggregate: 99.0,
  });
  assert.equal(gpuVramUsedIsPerDevice(gpu.devices ?? []), true);
  assert.equal(resolveGpuVramUsedGb(gpu), 40.5);
});

test("a partially attributed pair falls back to the aggregate, not a short sum", () => {
  // 40 GiB is capacity-forced onto the W7900; the idle card's 0.5 GiB is not.
  // Summing the known half alone would under-report the tile by that card.
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
  // A fabricated 0 used / full free is exactly what #7072 reported.
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
