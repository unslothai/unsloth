// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useRef, useState } from "react";

export interface GpuUtilizationDevice {
  index: number;
  index_kind: string | null;
  visible_ordinal: number | null;
  name: string | null;
  gpu_utilization_pct: number | null;
  temperature_c: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  vram_utilization_pct: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  power_utilization_pct: number | null;
}

export interface GpuUtilization {
  available: boolean;
  backend: string | null;
  backend_cuda_visible_devices: string | null;
  parent_visible_gpu_ids: number[];
  index_kind: string | null;
  devices: GpuUtilizationDevice[];
  gpu_utilization_pct: number | null;
  temperature_c: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  vram_utilization_pct: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  power_utilization_pct: number | null;
}

const DEFAULT_DEVICE: GpuUtilizationDevice = {
  index: 0,
  index_kind: null,
  visible_ordinal: null,
  name: null,
  gpu_utilization_pct: null,
  temperature_c: null,
  vram_used_gb: null,
  vram_total_gb: null,
  vram_utilization_pct: null,
  power_draw_w: null,
  power_limit_w: null,
  power_utilization_pct: null,
};

const DEFAULT: GpuUtilization = {
  available: false,
  backend: null,
  backend_cuda_visible_devices: null,
  parent_visible_gpu_ids: [],
  index_kind: null,
  devices: [],
  gpu_utilization_pct: null,
  temperature_c: null,
  vram_used_gb: null,
  vram_total_gb: null,
  vram_utilization_pct: null,
  power_draw_w: null,
  power_limit_w: null,
  power_utilization_pct: null,
};

function toNumberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeDevice(
  device: unknown,
  fallbackIndex: number,
): GpuUtilizationDevice | null {
  if (!device || typeof device !== "object") {
    return null;
  }

  const entry = device as Record<string, unknown>;
  return {
    ...DEFAULT_DEVICE,
    index: typeof entry.index === "number" ? entry.index : fallbackIndex,
    index_kind: typeof entry.index_kind === "string" ? entry.index_kind : null,
    visible_ordinal:
      typeof entry.visible_ordinal === "number" ? entry.visible_ordinal : null,
    name: typeof entry.name === "string" ? entry.name : null,
    gpu_utilization_pct: toNumberOrNull(entry.gpu_utilization_pct),
    temperature_c: toNumberOrNull(entry.temperature_c),
    vram_used_gb: toNumberOrNull(entry.vram_used_gb),
    vram_total_gb: toNumberOrNull(entry.vram_total_gb),
    vram_utilization_pct: toNumberOrNull(entry.vram_utilization_pct),
    power_draw_w: toNumberOrNull(entry.power_draw_w),
    power_limit_w: toNumberOrNull(entry.power_limit_w),
    power_utilization_pct: toNumberOrNull(entry.power_utilization_pct),
  };
}

function normalizeGpuUtilization(payload: unknown): GpuUtilization {
  if (!payload || typeof payload !== "object") {
    return DEFAULT;
  }

  const record = payload as Record<string, unknown>;
  const rawDevices = Array.isArray(record.devices)
    ? record.devices
    : Array.isArray(record.gpus)
      ? record.gpus
      : [];
  const devices = rawDevices
    .map((device, index) => normalizeDevice(device, index))
    .filter((device): device is GpuUtilizationDevice => device !== null);

  const normalized: GpuUtilization = {
    ...DEFAULT,
    available: record.available === true,
    backend: typeof record.backend === "string" ? record.backend : null,
    backend_cuda_visible_devices:
      typeof record.backend_cuda_visible_devices === "string"
        ? record.backend_cuda_visible_devices
        : null,
    parent_visible_gpu_ids: Array.isArray(record.parent_visible_gpu_ids)
      ? record.parent_visible_gpu_ids.filter(
          (gpuId): gpuId is number =>
            typeof gpuId === "number" && Number.isFinite(gpuId),
        )
      : [],
    index_kind:
      typeof record.index_kind === "string" ? record.index_kind : null,
    devices,
    gpu_utilization_pct: toNumberOrNull(record.gpu_utilization_pct),
    temperature_c: toNumberOrNull(record.temperature_c),
    vram_used_gb: toNumberOrNull(record.vram_used_gb),
    vram_total_gb: toNumberOrNull(record.vram_total_gb),
    vram_utilization_pct: toNumberOrNull(record.vram_utilization_pct),
    power_draw_w: toNumberOrNull(record.power_draw_w),
    power_limit_w: toNumberOrNull(record.power_limit_w),
    power_utilization_pct: toNumberOrNull(record.power_utilization_pct),
  };

  const hasLegacyMetrics = [
    normalized.gpu_utilization_pct,
    normalized.temperature_c,
    normalized.vram_used_gb,
    normalized.power_draw_w,
  ].some((value) => value != null);

  if (
    normalized.devices.length === 0 &&
    (normalized.available || hasLegacyMetrics)
  ) {
    normalized.devices = [
      {
        ...DEFAULT_DEVICE,
        index: 0,
        gpu_utilization_pct: normalized.gpu_utilization_pct,
        temperature_c: normalized.temperature_c,
        vram_used_gb: normalized.vram_used_gb,
        vram_total_gb: normalized.vram_total_gb,
        vram_utilization_pct: normalized.vram_utilization_pct,
        power_draw_w: normalized.power_draw_w,
        power_limit_w: normalized.power_limit_w,
        power_utilization_pct: normalized.power_utilization_pct,
      },
    ];
  }

  return normalized;
}

async function fetchGpuUtilization(
  endpoint: string,
): Promise<GpuUtilization | null> {
  const response = await authFetch(endpoint);
  if (!response.ok) {
    return null;
  }
  return normalizeGpuUtilization(await response.json());
}

/**
 * Poll `GET /api/train/hardware/visible` for live GPU utilization stats.
 *
 * Only polls while `enabled` is true (i.e. training is running).
 * Polling interval defaults to 10 000 ms.
 */
export function useGpuUtilization(
  enabled: boolean,
  intervalMs = 10_000,
): GpuUtilization {
  const [data, setData] = useState<GpuUtilization>(DEFAULT);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!enabled) {
      // Reset when training stops so the cards show "--" again
      setData(DEFAULT);
      return;
    }

    let cancelled = false;

    async function poll() {
      try {
        const payload =
          (await fetchGpuUtilization("/api/train/hardware/visible")) ??
          (await fetchGpuUtilization("/api/train/hardware"));
        if (!cancelled && payload) {
          setData(payload);
        }
      } catch {
        // Silently ignore — next poll will retry
      }
    }

    // Fetch immediately, then set up interval
    void poll();
    timerRef.current = setInterval(() => void poll(), intervalMs);

    return () => {
      cancelled = true;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [enabled, intervalMs]);

  return data;
}
