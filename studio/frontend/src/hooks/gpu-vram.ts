// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Split out of use-system.ts so the VRAM rules can be unit tested without pulling
// the auth and React graph in behind them. Typed structurally, so SystemGpuInfo and
// GpuDevice satisfy these without importing them.

export interface VramReportingDevice {
  vram_used_gb?: number;
}

export interface MemoryTotalDevice {
  memory_total_gb?: number;
  /** True when the reported GPU budget comes from shared system memory. */
  shared_memory?: boolean;
}

export interface VramReportingGpu {
  devices?: VramReportingDevice[];
  /** Used VRAM across the visible GPUs when no single device's usage could be
   * attributed. Windows ROCm only; null everywhere else. See #7452. */
  vram_used_gb_aggregate?: number | null;
}

/** Sum dedicated VRAM while counting a shared host-memory pool only once.
 *
 * Devices arrive rounded to 2dp, so summing them reintroduces float error
 * (three B200s at 179.06 give 537.1800000000001) and not every caller rounds
 * again before printing. Round back to the precision they arrived with. */
export function aggregateGpuMemoryTotalGb(
  devices: MemoryTotalDevice[],
): number {
  const dedicated = devices
    .filter((device) => !device.shared_memory)
    .reduce((sum, device) => sum + (device.memory_total_gb ?? 0), 0);
  const shared = Math.max(
    0,
    ...devices
      .filter((device) => device.shared_memory)
      .map((device) => device.memory_total_gb ?? 0),
  );
  return Math.round((dedicated + shared) * 100) / 100;
}

/** Whether every device reports its own usage, so each row and their sum are real. */
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
 * Per-device usage is preferred. On Windows ROCm nothing keys the LUID usage
 * counters to torch ordinals, so a usage that fits more than one card cannot be
 * attributed to either and every device reports unknown -- which is idle and every
 * small model on an asymmetric pair. The sum does not depend on that attribution,
 * so the backend still reports it, and rendering Unknown for a figure it already
 * has is what #7452 was.
 *
 * Never falls back to 0: a fabricated 0 used / full free is the #7072 symptom. */
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
