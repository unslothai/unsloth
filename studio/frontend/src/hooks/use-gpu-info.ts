// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";

export interface GpuInfo {
  available: boolean;
  name: string;
  memoryTotalGb: number;
  systemRamAvailableGb: number;
}

const DEFAULT_GPU: GpuInfo = {
  available: false,
  name: "Unknown",
  memoryTotalGb: 0,
  systemRamAvailableGb: 0,
};

// Module-level cache so multiple components share one fetch.
let cachedGpu: GpuInfo | null = null;
let fetchPromise: Promise<GpuInfo> | null = null;

async function fetchGpuOnce(): Promise<GpuInfo> {
  if (cachedGpu) return cachedGpu;
  if (fetchPromise) return fetchPromise;

  fetchPromise = (async () => {
    try {
      const res = await authFetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const gpuData = data?.gpu;
      if (!gpuData?.available || !gpuData.devices?.length) return DEFAULT_GPU;
      const devices = gpuData.devices as Array<{ name?: string; memory_total_gb?: number }>;
      const totalGb = devices.reduce((sum, d) => sum + (d.memory_total_gb ?? 0), 0);
      const info: GpuInfo = {
        available: true,
        name: devices[0]?.name ?? "Unknown",
        memoryTotalGb: totalGb,
        systemRamAvailableGb: data?.memory?.available_gb ?? 0,
      };
      cachedGpu = info;
      return info;
    } catch {
      // Reset promise so subsequent calls retry (e.g. backend wasn't ready)
      fetchPromise = null;
      return DEFAULT_GPU;
    }
  })();

  return fetchPromise;
}

/**
 * Fetch GPU info from /api/system. Cached at module level, so only one request
 * is made no matter how many components call this hook.
 */
export function useGpuInfo(): GpuInfo {
  const [gpu, setGpu] = useState<GpuInfo>(cachedGpu ?? DEFAULT_GPU);

  useEffect(() => {
    if (cachedGpu) return;

    let cancelled = false;
    fetchGpuOnce().then((info) => {
      if (!cancelled) setGpu(info);
    });
    return () => { cancelled = true; };
  }, []);

  return gpu;
}

export interface GpuDevice {
  index: number;
  name: string;
  memoryTotalGb: number;
}

let cachedDevices: GpuDevice[] | null = null;
let devicesPromise: Promise<GpuDevice[]> | null = null;

async function fetchGpuDevicesOnce(): Promise<GpuDevice[]> {
  if (cachedDevices) return cachedDevices;
  if (devicesPromise) return devicesPromise;
  devicesPromise = (async () => {
    try {
      const res = await authFetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const devices = (data?.gpu?.devices ?? []) as Array<{
        index?: number;
        name?: string;
        memory_total_gb?: number;
      }>;
      const list = devices
        .filter((d) => typeof d.index === "number")
        .map((d) => ({
          index: d.index as number,
          name: d.name ?? `GPU ${d.index}`,
          memoryTotalGb: d.memory_total_gb ?? 0,
        }));
      cachedDevices = list;
      return list;
    } catch {
      devicesPromise = null;
      return [];
    }
  })();
  return devicesPromise;
}

/** All backend-visible GPUs (index, name, total VRAM). Cached module-wide. */
export function useGpuDevices(): GpuDevice[] {
  const [devices, setDevices] = useState<GpuDevice[]>(cachedDevices ?? []);
  useEffect(() => {
    if (cachedDevices) return;
    let cancelled = false;
    fetchGpuDevicesOnce().then((d) => {
      if (!cancelled) setDevices(d);
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return devices;
}
