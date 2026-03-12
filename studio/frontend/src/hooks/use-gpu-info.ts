// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

export interface GpuInfo {
  available: boolean;
  name: string;
  memoryTotalGb: number;
}

const DEFAULT_GPU: GpuInfo = {
  available: false,
  name: "Unknown",
  memoryTotalGb: 0,
};

// Module-level cache so multiple components share one fetch.
let cachedGpu: GpuInfo | null = null;
let fetchPromise: Promise<GpuInfo> | null = null;

async function fetchGpuOnce(): Promise<GpuInfo> {
  if (cachedGpu) return cachedGpu;
  if (fetchPromise) return fetchPromise;

  fetchPromise = (async () => {
    try {
      const res = await fetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const gpuData = data?.gpu;
      if (!gpuData?.available || !gpuData.devices?.length) return DEFAULT_GPU;
      const dev = gpuData.devices[0];
      const info: GpuInfo = {
        available: true,
        name: dev.name ?? "Unknown",
        memoryTotalGb: dev.memory_total_gb ?? 0,
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
 * Fetch GPU info from the backend /api/system endpoint.
 *
 * The result is cached at module level -- only one network request is made
 * regardless of how many components call this hook.
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
