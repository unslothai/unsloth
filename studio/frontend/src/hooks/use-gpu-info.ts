// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import type { SystemInfoResponse } from "./use-system";

export interface GpuInfo {
  available: boolean;
  name: string;
  memoryTotalGb: number;
  cpuCore: number;
  cpuThread: number;
  systemRamAvailableGb: number;
  systemRamTotalGb: number
}

export interface SystemGpuDevice {
  index: number;
  indexKind: GpuIndexKind | null;
  name: string;
  memoryTotalGb: number;
  /** Free VRAM at fetch time, or total VRAM when usage is unavailable. */
  memoryFreeGb: number;
  /** Whether `index` is safe to send as gpu_ids. */
  pinnable: boolean;
}

export type GpuIndexKind = "physical" | "vulkan";

const DEFAULT_GPU: GpuInfo = {
  available: false,
  name: "Unknown",
  memoryTotalGb: 0,
  cpuCore: 0,
  cpuThread: 0,
  systemRamAvailableGb: 0,
  systemRamTotalGb: 0
};

// One module-level cache so every GPU hook shares a single /api/system fetch.
let cachedSystem: SystemInfoResponse | null = null;
let systemPromise: Promise<SystemInfoResponse | null> | null = null;

async function fetchSystemOnce(): Promise<SystemInfoResponse | null> {
  if (cachedSystem) return cachedSystem;
  if (systemPromise) return systemPromise;
  systemPromise = (async () => {
    try {
      const res = await authFetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      cachedSystem = (await res.json()) as SystemInfoResponse;
      return cachedSystem;
    } catch {
      systemPromise = null; // reset so a later call retries (backend not ready)
      return null;
    }
  })();
  return systemPromise;
}

function toGpuInfo(data: SystemInfoResponse | null): GpuInfo {
  // CPU/RAM exist even on GPU-less hosts (e.g. Mac), so populate them on every
  // path: unified-memory math still needs a RAM budget to work with.
  const base = {
    cpuCore: data?.cpu?.physical_count ?? 0,
    cpuThread: data?.cpu?.logical_count ?? 0,
    systemRamAvailableGb: data?.memory?.available_gb ?? 0,
    systemRamTotalGb: data?.memory?.total_gb ?? 0,
  };
  const gpuData = data?.gpu;
  const devices = gpuData?.devices ?? [];
  if (!gpuData?.available || !devices.length) {
    return { ...DEFAULT_GPU, ...base };
  }
  return {
    ...base,
    available: true,
    name: devices[0]?.name ?? "Unknown",
    memoryTotalGb: devices.reduce((sum, d) => sum + (d.memory_total_gb ?? 0), 0),
  };
}

function toGpuDevices(data: SystemInfoResponse | null): SystemGpuDevice[] {
  // The backend declares whether its device indices are pinnable.
  const pinnableBackend = data?.gpu?.gguf_gpu_ids_supported !== false;
  // A llama.cpp build that pins in its own index space (Vulkan ordinals) publishes
  // those candidates separately, so gpu.devices can keep describing the PyTorch
  // accelerators that useGpuInfo() sums for fit badges and training sizing. Older
  // backends omit the key and pin in the gpu.devices space.
  const pinCandidates = data?.gpu?.gguf_gpu_devices?.length
    ? data.gpu.gguf_gpu_devices
    : data?.gpu?.devices;
  return (pinCandidates ?? [])
    .filter((d) => typeof d.index === "number")
    .map((d) => ({
      index: d.index as number,
      indexKind:
        d.index_kind === "physical" || d.index_kind === "vulkan"
          ? d.index_kind
          : null,
      name: d.name ?? `GPU ${d.index}`,
      memoryTotalGb: d.memory_total_gb ?? 0,
      memoryFreeGb: d.vram_free_gb ?? 0,
      pinnable:
        pinnableBackend &&
        (d.index_kind === "physical" || d.index_kind === "vulkan"),
    }));
}

/** Aggregate GPU info from /api/system; shares one module-level fetch across all GPU hooks. */
export function useGpuInfo(): GpuInfo {
  const [gpu, setGpu] = useState<GpuInfo>(
    cachedSystem ? toGpuInfo(cachedSystem) : DEFAULT_GPU,
  );
  useEffect(() => {
    // No early return on cachedSystem: a consumer mounting as the cache fills
    // (between render and effect) would otherwise stay stuck at the default.
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
    // No early return on cachedSystem: a consumer mounting as the cache fills
    // (between render and effect) would otherwise stay stuck at the default.
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

/** Warm the shared system cache before validating persisted GPU IDs. */
export async function ensureGpuDeviceCache(): Promise<void> {
  await fetchSystemOnce();
}

/** Cached pinnable IDs, null before fetch, or [] when pinning is unavailable. */
export function cachedPinnableGpuIndices(): number[] | null {
  if (!cachedSystem) return null;
  const pinnable = toGpuDevices(cachedSystem).filter((d) => d.pinnable);
  return pinnable.length > 1 ? pinnable.map((d) => d.index) : [];
}

/** Cached index namespace, undefined before fetch and null when unavailable. */
export function cachedPinnableGpuIndexKind():
  | GpuIndexKind
  | null
  | undefined {
  if (!cachedSystem) return undefined;
  const pinnable = toGpuDevices(cachedSystem).filter((d) => d.pinnable);
  const kinds = new Set(pinnable.map((d) => d.indexKind).filter((k) => k));
  return pinnable.length > 1 && kinds.size === 1
    ? ([...kinds][0] as GpuIndexKind)
    : null;
}
