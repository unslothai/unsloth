// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sharesHostMemoryDevice } from "./gpu-vram.ts";
import type { GpuDevice, SystemGpuInfo } from "./use-system";

export function hasSharedMemoryHeadroom(device: GpuDevice): boolean {
  return device.index_kind === "vulkan" && device.shared_memory === true;
}

/** A usage device whose whole budget is host RAM (Apple unified memory, CUDA integrated): its free reading is headroom. */
export function hasUnifiedPoolHeadroom(device: GpuDevice): boolean {
  const total = device.memory_total_gb ?? 0;
  const hostBacked = device.shared_memory_host_backed_gb;
  return (
    !hasSharedMemoryHeadroom(device) &&
    device.shared_memory === true &&
    total > 0 &&
    Number.isFinite(hostBacked) &&
    (hostBacked as number) >= total &&
    Number.isFinite(device.vram_free_gb) &&
    (device.vram_free_gb as number) >= 0
  );
}

export function sharedMemoryAvailableGb(devices: GpuDevice[]): number | null {
  if (
    !devices.length ||
    devices.some(
      (device) =>
        !Number.isFinite(device.vram_free_gb) ||
        (device.vram_free_gb as number) < 0,
    )
  )
    return null;
  // Multiple iGPUs are views of the same host pool, not additive capacity.
  return Math.max(...devices.map((device) => device.vram_free_gb as number));
}

/** Shared-RAM detail for the VRAM tile; drops the estimate line when no free reading exists. */
export function sharedMemoryDetail(
  availableGb: number | null,
  label: string,
  estimate: (gb: number) => string,
): string {
  return availableGb === null ? label : `${label}\n${estimate(availableGb)}`;
}

/** A shared iGPU with no usage counter beside cards that all report one: shown as shared RAM rather
 * than blanking the cards' used VRAM, or counting its pool as free. */
function parkedSharedDevices(
  devices: GpuDevice[],
  aggregate: number | null | undefined,
): GpuDevice[] {
  if (Number.isFinite(aggregate)) return [];
  const rest = devices.filter((device) => !hasSharedMemoryHeadroom(device));
  const unread = rest.filter((device) => !Number.isFinite(device.vram_used_gb));
  return unread.length > 0 &&
    unread.every(sharesHostMemoryDevice) &&
    rest.some((device) => !sharesHostMemoryDevice(device))
    ? unread
    : [];
}

/** Presentation only. Keep allocation budgets and model-fit capacity unchanged. */
export function gpuMemoryDisplay(gpu: SystemGpuInfo | null | undefined) {
  const devices = gpu?.devices ?? [];
  const parked = parkedSharedDevices(devices, gpu?.vram_used_gb_aggregate);
  const sharedDevices = devices.filter(
    (device) => hasSharedMemoryHeadroom(device) || parked.includes(device),
  );
  const usageDevices = devices.filter(
    (device) => !sharedDevices.includes(device),
  );
  return {
    sharedDevices,
    usageDevices,
    sharedOnly: sharedDevices.length > 0 && usageDevices.length === 0,
    sharedAvailableGb: sharedMemoryAvailableGb(sharedDevices),
    unifiedAvailableGb: sharedMemoryAvailableGb(
      usageDevices.filter(hasUnifiedPoolHeadroom),
    ),
    usageGpu: gpu
      ? {
          ...gpu,
          devices: usageDevices,
          // A host aggregate cannot be attributed to a filtered subset of its devices.
          vram_used_gb_aggregate: sharedDevices.length
            ? null
            : gpu.vram_used_gb_aggregate,
        }
      : gpu,
  };
}
