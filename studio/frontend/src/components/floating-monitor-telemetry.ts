// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GpuUtilization } from "@/hooks/use-gpu-utilization";
import type { GpuDevice } from "@/hooks/use-system";

export type FloatingMonitorGpuTelemetry = GpuDevice & {
  temperature_c: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
};

export function hasFloatingMonitorGpuTelemetry(
  devices: readonly FloatingMonitorGpuTelemetry[],
): boolean {
  return devices.some(
    (device) =>
      device.temperature_c != null || device.power_draw_w != null,
  );
}

export function resolveFloatingMonitorGpuTelemetry(
  devices: readonly GpuDevice[],
  displayedBackend: string | undefined,
  gpuUtilization: GpuUtilization,
): FloatingMonitorGpuTelemetry[] {
  const utilizationDevices =
    gpuUtilization.devices && gpuUtilization.devices.length > 0
      ? gpuUtilization.devices
      : gpuUtilization.available
        ? [gpuUtilization]
        : [];
  const backendsMatch =
    !displayedBackend ||
    !gpuUtilization.backend ||
    displayedBackend === gpuUtilization.backend;

  return devices.map((device) => {
    const telemetry = backendsMatch
      ? (utilizationDevices.find(
          (candidate) =>
            candidate.index != null && candidate.index === device.index,
        ) ??
        utilizationDevices.find(
          (candidate) =>
            candidate.visible_ordinal != null &&
            candidate.visible_ordinal === device.visible_ordinal,
        ) ??
        (devices.length === 1 && utilizationDevices.length === 1
          ? utilizationDevices[0]
          : undefined))
      : undefined;

    return {
      ...device,
      temperature_c: telemetry?.temperature_c ?? null,
      power_draw_w: telemetry?.power_draw_w ?? null,
      power_limit_w: telemetry?.power_limit_w ?? null,
    };
  });
}
