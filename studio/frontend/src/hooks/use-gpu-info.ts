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
  /** Free VRAM at fetch time. Degrades to the total when the utilization
   * probe had no usage data; 0 only when the total is unknown too. */
  memoryFreeGb: number;
  /** True when `index` is safe to send as gpu_ids. This covers CUDA/HIP
   *  physical IDs and ggml Vulkan ordinals, but not unresolved relative IDs. */
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
  // The backend owns the index contract. CUDA/HIP expose stable physical IDs,
  // while Vulkan exposes compact ggml ordinals. XPU and unresolved relative
  // masks report support=false. Missing support info preserves compatibility
  // with older backends.
  const pinnableBackend = data?.gpu?.gguf_gpu_ids_supported !== false;
  return (data?.gpu?.devices ?? [])
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

/**
 * Await the shared /api/system fetch so cachedPinnableGpuIndices (and the
 * store's reconcilePersistedGpuIds) can validate a persisted pick before a
 * load path sends it -- on a cold cache the reconcile passes ids through
 * unvalidated, and a stale cross-host pick then fails /load with the picker
 * hidden. Resolves immediately once the module cache is warm; a failed fetch
 * keeps the cache cold, preserving the "can't validate, backend guards"
 * degradation.
 */
export async function ensureGpuDeviceCache(): Promise<void> {
  await fetchSystemOnce();
}

/**
 * Pinnable physical GPU indices from the already-fetched /api/system cache, for
 * non-React code (the store) that needs to validate a persisted `gpu_ids` pick
 * without triggering a fetch. Returns:
 *  - `null` when the cache isn't populated yet (caller can't validate, so keep
 *    the pick and let the backend guard reject a truly bad one);
 *  - `[]` when the host has no pinnable multi-GPU set (single GPU, or relative/
 *    UUID-masked indices) -- the picker is hidden, so any saved pick is stale;
 *  - the physical indices otherwise.
 */
export function cachedPinnableGpuIndices(): number[] | null {
  if (!cachedSystem) return null;
  const pinnable = toGpuDevices(cachedSystem).filter((d) => d.pinnable);
  // Mirrors the sheet's showGpuPicker gate: only a 2+ pinnable-GPU host can pin.
  return pinnable.length > 1 ? pinnable.map((d) => d.index) : [];
}

/** Index namespace for persisted gpu_ids. Undefined means the cache is cold;
 * null means the current host has no single pinnable namespace. */
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
