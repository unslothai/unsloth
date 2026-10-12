// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GpuDevice } from "../hooks/use-system.ts";

export const MONITOR_DETAIL_METRICS = [
  "cpuLoad",
  "gpuLoad",
  "temperature",
  "power",
] as const;

export type MonitorDetailMetric = (typeof MONITOR_DETAIL_METRICS)[number];

/** An empty list is a choice; anything else with no known metric (older state, hand edits) means all. */
export function selectedDetailMetrics(
  stored: unknown,
): ReadonlySet<MonitorDetailMetric> {
  const known = Array.isArray(stored)
    ? MONITOR_DETAIL_METRICS.filter((metric) => stored.includes(metric))
    : [];
  return new Set(
    known.length > 0 || (Array.isArray(stored) && stored.length === 0)
      ? known
      : MONITOR_DETAIL_METRICS,
  );
}

export interface GpuTelemetryRow {
  key: string;
  name: string;
  loadPct: number | null;
  temperatureC: number | null;
  powerDrawW: number | null;
  powerLimitW: number | null;
  powerPct: number | null;
}

function finite(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** One row per device that reported at least one selected reading; nothing is fabricated. */
export function gpuTelemetryRows(
  devices: readonly GpuDevice[],
  metrics: ReadonlySet<MonitorDetailMetric>,
): GpuTelemetryRow[] {
  const rows: GpuTelemetryRow[] = [];
  devices.forEach((device, position) => {
    const ordinal = device.visible_ordinal ?? device.index ?? position;
    const loadPct = metrics.has("gpuLoad")
      ? finite(device.gpu_utilization_pct)
      : null;
    const temperatureC = metrics.has("temperature")
      ? finite(device.temperature_c)
      : null;
    const powerDrawW = metrics.has("power")
      ? finite(device.power_draw_w)
      : null;
    const powerLimitW =
      powerDrawW !== null ? finite(device.power_limit_w) : null;
    if (loadPct === null && temperatureC === null && powerDrawW === null) {
      return;
    }
    rows.push({
      key: `${ordinal}`,
      name:
        devices.length > 1
          ? `GPU ${ordinal}${device.name ? `: ${device.name}` : ""}`
          : (device.name ?? `GPU ${ordinal}`),
      loadPct,
      temperatureC,
      powerDrawW,
      powerLimitW,
      powerPct:
        powerDrawW !== null && powerLimitW !== null && powerLimitW > 0
          ? (powerDrawW / powerLimitW) * 100
          : null,
    });
  });
  return rows;
}

/** Most cards throttle in the high 80s, so the bar warns before that. */
export function temperatureLevel(celsius: number): "ok" | "warm" | "hot" {
  if (celsius >= 90) {
    return "hot";
  }
  if (celsius >= 80) {
    return "warm";
  }
  return "ok";
}

export function formatTemperature(celsius: number): string {
  return `${Math.round(celsius)}°C`;
}

export function formatPower(drawW: number, limitW: number | null): string {
  return limitW !== null && limitW > 0
    ? `${Math.round(drawW)} / ${Math.round(limitW)} W`
    : `${Math.round(drawW)} W`;
}
