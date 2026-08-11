// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Split out of use-system.ts so the VRAM reading rules can be unit tested without
// pulling the auth and React graph in behind them. Typed structurally, so
// SystemGpuInfo and GpuDevice satisfy these without importing them.

export interface VramReportingDevice {
  vram_used_gb?: number;
}

export interface VramReportingGpu {
  devices?: VramReportingDevice[];
  /** Used VRAM across the visible GPUs when no single device's usage could be
   * attributed. Windows ROCm only; null everywhere else. See #7452. */
  vram_used_gb_aggregate?: number | null;
}

/** Whether every device reports its own usage, so the per-device figures can be
 * summed and each row can show a real number. */
export function gpuVramUsedIsPerDevice(
  devices: VramReportingDevice[],
): boolean {
  return (
    devices.length > 0 &&
    devices.every((device) => Number.isFinite(device.vram_used_gb))
  );
}

/** Used VRAM across the GPUs, or null when it is genuinely unknown.
 *
 * Per-device usage is the preferred source. On Windows ROCm there is no shared
 * key between the LUID usage counters and torch ordinals, so a usage that fits
 * more than one card cannot be attributed to either and every device reports
 * unknown -- which is idle and every small model on an asymmetric pair. The sum
 * does not depend on that attribution, so the backend still reports it, and
 * rendering Unknown for a figure it already has is what #7452 was.
 *
 * Never falls back to 0: a fabricated 0 used / full free is the #7072 symptom
 * this pair started from. */
export function resolveGpuVramUsedGb(
  gpu: VramReportingGpu | null | undefined,
): number | null {
  const devices = gpu?.devices ?? [];
  if (gpuVramUsedIsPerDevice(devices)) {
    return devices.reduce((sum, device) => sum + (device.vram_used_gb ?? 0), 0);
  }
  const aggregate = gpu?.vram_used_gb_aggregate;
  return Number.isFinite(aggregate) ? (aggregate as number) : null;
}
