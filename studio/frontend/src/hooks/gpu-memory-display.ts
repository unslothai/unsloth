// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GpuDevice, SystemGpuInfo } from "./use-system";

/** Vulkan shared devices report allocation headroom, not a fixed VRAM pool. */
export function hasSharedMemoryHeadroom(device: GpuDevice): boolean {
  return device.index_kind === "vulkan" && device.shared_memory === true;
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

/** Presentation only. Keep allocation budgets and model-fit capacity unchanged. */
export function gpuMemoryDisplay(gpu: SystemGpuInfo | null | undefined) {
  const devices = gpu?.devices ?? [];
  const sharedDevices = devices.filter(hasSharedMemoryHeadroom);
  const usageDevices = devices.filter(
    (device) => !hasSharedMemoryHeadroom(device),
  );
  return {
    sharedDevices,
    usageDevices,
    sharedOnly: sharedDevices.length > 0 && usageDevices.length === 0,
    sharedAvailableGb: sharedMemoryAvailableGb(sharedDevices),
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
