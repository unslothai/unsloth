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
      const ramAvailableGb = data?.memory?.available_gb ?? 0;
      const gpuData = data?.gpu;
      if (!gpuData?.available || !gpuData.devices?.length) {
        // No discrete GPU (e.g. Mac): still surface system RAM so memory math
        // (unified memory) has a budget to work with.
        const info: GpuInfo = { ...DEFAULT_GPU, systemRamAvailableGb: ramAvailableGb };
        cachedGpu = info;
        return info;
      }
      const devices = gpuData.devices as Array<{ name?: string; memory_total_gb?: number }>;
      const totalGb = devices.reduce((sum, d) => sum + (d.memory_total_gb ?? 0), 0);
      const info: GpuInfo = {
        available: true,
        name: devices[0]?.name ?? "Unknown",
        memoryTotalGb: totalGb,
        systemRamAvailableGb: ramAvailableGb,
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
