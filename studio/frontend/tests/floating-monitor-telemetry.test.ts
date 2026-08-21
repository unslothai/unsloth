// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  hasFloatingMonitorGpuTelemetry,
  resolveFloatingMonitorGpuTelemetry,
} from "../src/components/floating-monitor-telemetry.ts";
import type { GpuUtilization } from "../src/hooks/use-gpu-utilization.ts";

function utilization(
  overrides: Partial<GpuUtilization> = {},
): GpuUtilization {
  return {
    available: true,
    backend: "cuda",
    gpu_utilization_pct: null,
    temperature_c: null,
    vram_used_gb: null,
    vram_total_gb: null,
    vram_utilization_pct: null,
    power_draw_w: null,
    power_limit_w: null,
    power_utilization_pct: null,
    ...overrides,
  };
}

test("single-GPU telemetry uses the legacy top-level payload", () => {
  const resolved = resolveFloatingMonitorGpuTelemetry(
    [{ index: 0, name: "Apple M4" }],
    "mlx",
    utilization({
      backend: "mlx",
      temperature_c: 58.9,
      power_draw_w: 0.2,
    }),
  );

  assert.deepEqual(resolved, [
    {
      index: 0,
      name: "Apple M4",
      temperature_c: 58.9,
      power_draw_w: 0.2,
      power_limit_w: null,
    },
  ]);
});

test("multi-GPU telemetry follows physical indices instead of response order", () => {
  const resolved = resolveFloatingMonitorGpuTelemetry(
    [
      { index: 5, visible_ordinal: 0, name: "GPU Five" },
      { index: 3, visible_ordinal: 1, name: "GPU Three" },
    ],
    "cuda",
    utilization({
      devices: [
        utilization({
          index: 3,
          visible_ordinal: 1,
          temperature_c: 62,
          power_draw_w: 180,
          power_limit_w: 300,
        }),
        utilization({
          index: 5,
          visible_ordinal: 0,
          temperature_c: 54,
          power_draw_w: 120,
          power_limit_w: 300,
        }),
      ],
    }),
  );

  assert.deepEqual(
    resolved.map(({ temperature_c, power_draw_w }) => ({
      temperature_c,
      power_draw_w,
    })),
    [
      { temperature_c: 54, power_draw_w: 120 },
      { temperature_c: 62, power_draw_w: 180 },
    ],
  );
});

test("visible ordinals support relative-index telemetry", () => {
  const [resolved] = resolveFloatingMonitorGpuTelemetry(
    [{ visible_ordinal: 1, name: "Relative GPU" }],
    "xpu",
    utilization({
      backend: "xpu",
      devices: [
        utilization({
          backend: "xpu",
          visible_ordinal: 1,
          temperature_c: 48,
        }),
      ],
    }),
  );

  assert.equal(resolved.temperature_c, 48);
  assert.equal(resolved.power_draw_w, null);
});

test("backend mismatches never merge unrelated index spaces", () => {
  const [resolved] = resolveFloatingMonitorGpuTelemetry(
    [{ index: 0, name: "Vulkan GPU" }],
    "vulkan",
    utilization({
      backend: "cuda",
      index: 0,
      temperature_c: 90,
      power_draw_w: 400,
    }),
  );

  assert.equal(resolved.temperature_c, null);
  assert.equal(resolved.power_draw_w, null);
});

test("unavailable telemetry remains unknown", () => {
  const [resolved] = resolveFloatingMonitorGpuTelemetry(
    [{ index: 0, name: "GPU" }],
    "cuda",
    utilization({ available: false }),
  );

  assert.equal(resolved.temperature_c, null);
  assert.equal(resolved.power_draw_w, null);
  assert.equal(resolved.power_limit_w, null);
});

test("a power limit without live readings stays hidden", () => {
  const resolved = resolveFloatingMonitorGpuTelemetry(
    [{ index: 0, name: "GPU" }],
    "cuda",
    utilization({ power_limit_w: 300 }),
  );

  assert.equal(hasFloatingMonitorGpuTelemetry(resolved), false);
});

test("temperature or power draw reveals telemetry", () => {
  const temperature = resolveFloatingMonitorGpuTelemetry(
    [{ index: 0, name: "GPU" }],
    "cuda",
    utilization({ temperature_c: 0 }),
  );
  const power = resolveFloatingMonitorGpuTelemetry(
    [{ index: 0, name: "GPU" }],
    "cuda",
    utilization({ power_draw_w: 0 }),
  );

  assert.equal(hasFloatingMonitorGpuTelemetry(temperature), true);
  assert.equal(hasFloatingMonitorGpuTelemetry(power), true);
});
