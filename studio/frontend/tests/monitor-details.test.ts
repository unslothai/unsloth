// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MONITOR_DETAIL_METRICS,
  formatPower,
  formatTemperature,
  gpuTelemetryRows,
  selectedDetailMetrics,
  temperatureLevel,
} from "../src/components/monitor-details.ts";

const ALL = selectedDetailMetrics(undefined);

test("stored metric selections survive and garbage falls back to every metric", () => {
  assert.deepEqual([...ALL], [...MONITOR_DETAIL_METRICS]);
  assert.deepEqual(
    [...selectedDetailMetrics("power")],
    [...MONITOR_DETAIL_METRICS],
  );
  assert.deepEqual(
    [...selectedDetailMetrics(["power", "bogus", "temperature"])],
    ["temperature", "power"],
  );
  assert.deepEqual([...selectedDetailMetrics([])], []);
  assert.deepEqual(
    [...selectedDetailMetrics(["bogus", 1])],
    [...MONITOR_DETAIL_METRICS],
  );
});

test("rows carry only readings the backend reported", () => {
  const rows = gpuTelemetryRows(
    [
      {
        index: 0,
        name: "NVIDIA B200",
        gpu_utilization_pct: 42,
        temperature_c: 61,
        power_draw_w: 312.5,
        power_limit_w: 1000,
      },
      { index: 1, name: "NVIDIA B200" },
      { index: 2, name: "NVIDIA B200", temperature_c: 55, power_draw_w: 80 },
    ],
    ALL,
  );
  assert.deepEqual(rows, [
    {
      key: "0",
      name: "GPU 0: NVIDIA B200",
      loadPct: 42,
      temperatureC: 61,
      powerDrawW: 312.5,
      powerLimitW: 1000,
      powerPct: 31.25,
    },
    {
      key: "2",
      name: "GPU 2: NVIDIA B200",
      loadPct: null,
      temperatureC: 55,
      powerDrawW: 80,
      powerLimitW: null,
      powerPct: null,
    },
  ]);
});

test("a deselected metric is dropped, and a device left with nothing is skipped", () => {
  const device = {
    index: 0,
    name: "RTX 4090",
    temperature_c: 70,
    gpu_utilization_pct: 10,
  };
  const rows = gpuTelemetryRows(
    [device],
    selectedDetailMetrics(["temperature"]),
  );
  assert.equal(rows.length, 1);
  assert.equal(rows[0].name, "RTX 4090");
  assert.equal(rows[0].loadPct, null);
  assert.equal(rows[0].temperatureC, 70);
  assert.deepEqual(
    gpuTelemetryRows([device], selectedDetailMetrics(["power"])),
    [],
  );
  assert.deepEqual(
    gpuTelemetryRows([{ index: 0, temperature_c: Number.NaN }], ALL),
    [],
  );
});

test("temperature turns amber at 80 C and red at 90 C", () => {
  assert.equal(temperatureLevel(79.9), "ok");
  assert.equal(temperatureLevel(80), "warm");
  assert.equal(temperatureLevel(90), "hot");
  assert.equal(formatTemperature(61.4), "61°C");
  assert.equal(formatPower(312.5, 1000), "313 / 1000 W");
  assert.equal(formatPower(80, null), "80 W");
});

test("the detailed view is opt-in and adds no request of its own", () => {
  const store = readFileSync(
    new URL(
      "../src/features/settings/stores/monitor-overlay-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(store, /showDetails: false,/);
  const monitor = readFileSync(
    new URL("../src/components/floating-monitor.tsx", import.meta.url),
    "utf8",
  );
  // Rows render from the /api/system poll the monitor already runs.
  assert.doesNotMatch(monitor, /useGpuUtilization|api\/train\/hardware/);
  assert.match(monitor, /showDetails && displayedGpu\?\.available/);
});
