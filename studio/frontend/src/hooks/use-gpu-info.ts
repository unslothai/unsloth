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

export interface SystemGpuDevice {
  index: number;
  name: string;
  memoryTotalGb: number;
}

const DEFAULT_GPU: GpuInfo = {
  available: false,
  name: "Unknown",
  memoryTotalGb: 0,
  systemRamAvailableGb: 0,
};

interface SystemPayload {
  gpu?: {
    available?: boolean;
    devices?: Array<{ index?: number; name?: string; memory_total_gb?: number }>;
  };
  memory?: { available_gb?: number };
}

// One module-level cache so every GPU hook shares a single /api/system fetch.
let cachedSystem: SystemPayload | null = null;
let systemPromise: Promise<SystemPayload | null> | null = null;

async function fetchSystemOnce(): Promise<SystemPayload | null> {
  if (cachedSystem) return cachedSystem;
  if (systemPromise) return systemPromise;
  systemPromise = (async () => {
    try {
      const res = await authFetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      cachedSystem = (await res.json()) as SystemPayload;
      return cachedSystem;
    } catch {
      systemPromise = null; // reset so a later call retries (backend not ready)
      return null;
    }
  })();
  return systemPromise;
}

function toGpuInfo(data: SystemPayload | null): GpuInfo {
  const gpuData = data?.gpu;
  if (!gpuData?.available || !gpuData.devices?.length) return DEFAULT_GPU;
  const devices = gpuData.devices;
  const totalGb = devices.reduce((sum, d) => sum + (d.memory_total_gb ?? 0), 0);
  return {
    available: true,
    name: devices[0]?.name ?? "Unknown",
    memoryTotalGb: totalGb,
    systemRamAvailableGb: data?.memory?.available_gb ?? 0,
  };
}

function toGpuDevices(data: SystemPayload | null): SystemGpuDevice[] {
  return (data?.gpu?.devices ?? [])
    .filter((d) => typeof d.index === "number")
    .map((d) => ({
      index: d.index as number,
      name: d.name ?? `GPU ${d.index}`,
      memoryTotalGb: d.memory_total_gb ?? 0,
    }));
}

/**
 * Aggregate GPU info from /api/system. Cached at module level, so only one
 * request is made no matter how many GPU hooks are mounted.
 */
export function useGpuInfo(): GpuInfo {
  const [gpu, setGpu] = useState<GpuInfo>(
    cachedSystem ? toGpuInfo(cachedSystem) : DEFAULT_GPU,
  );
  useEffect(() => {
    if (cachedSystem) return;
    let cancelled = false;
    fetchSystemOnce().then((d) => {
      if (!cancelled) setGpu(toGpuInfo(d));
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return gpu;
}

/** All backend-visible GPUs (index, name, total VRAM); shares the same fetch. */
export function useGpuDevices(): SystemGpuDevice[] {
  const [devices, setDevices] = useState<SystemGpuDevice[]>(
    cachedSystem ? toGpuDevices(cachedSystem) : [],
  );
  useEffect(() => {
    if (cachedSystem) return;
    let cancelled = false;
    fetchSystemOnce().then((d) => {
      if (!cancelled) setDevices(toGpuDevices(d));
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return devices;
}
