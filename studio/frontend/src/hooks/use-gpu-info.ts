// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import {
  aggregateGpuMemoryTotalGb,
  type SystemInfoResponse,
} from "./use-system";

export interface GpuInfo {
  available: boolean;
  budgetKnown: boolean;
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
  /** Whether the separate DiffusionGemma runner can use this physical ID. */
  diffusionPinnable: boolean;
}

export type GpuIndexKind = "physical" | "vulkan";

const DEFAULT_GPU: GpuInfo = {
  available: false,
  budgetKnown: false,
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
// An unavailable Vulkan probe answers gguf_devices as an empty list, so the
// picker starts hidden and only useInferenceGpuInfo retries. Without this,
// hooks that fetch once never see the recovery and stay hidden until remount.
const systemSubscribers = new Set<(data: SystemInfoResponse | null) => void>();

async function fetchSystemOnce(force = false): Promise<SystemInfoResponse | null> {
  if (!force && cachedSystem) return cachedSystem;
  if (systemPromise) return systemPromise;
  systemPromise = (async () => {
    try {
      const res = await authFetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      cachedSystem = (await res.json()) as SystemInfoResponse;
      systemSubscribers.forEach((notify) => notify(cachedSystem));
      return cachedSystem;
    } catch {
      return null;
    } finally {
      systemPromise = null;
    }
  })();
  return systemPromise;
}

function toGpuInfo(
  data: SystemInfoResponse | null,
  source: "gpu" | "inference_gpu" = "gpu",
): GpuInfo {
  // CPU/RAM exist even on GPU-less hosts (e.g. Mac), so populate them on every
  // path: unified-memory math still needs a RAM budget to work with.
  const base = {
    cpuCore: data?.cpu?.physical_count ?? 0,
    cpuThread: data?.cpu?.logical_count ?? 0,
    systemRamAvailableGb: data?.memory?.available_gb ?? 0,
    systemRamTotalGb: data?.memory?.total_gb ?? 0,
  };
  const gpuData =
    source === "inference_gpu"
      ? (data?.inference_gpu ?? data?.gpu)
      : data?.gpu;
  const devices = gpuData?.devices ?? [];
  if (!gpuData?.available || !devices.length) {
    return { ...DEFAULT_GPU, ...base, budgetKnown: data !== null };
  }
  return {
    ...base,
    // A Vulkan iGPU's reported budget is capped shared system RAM, not an
    // independent VRAM pool. Do not offer the same RAM again for CPU offload.
    systemRamAvailableGb: devices.some((device) => device.shared_memory)
      ? 0
      : base.systemRamAvailableGb,
    available: true,
    budgetKnown: true,
    name: devices[0]?.name ?? "Unknown",
    memoryTotalGb: aggregateGpuMemoryTotalGb(devices),
  };
}

function toGpuDevices(data: SystemInfoResponse | null): SystemGpuDevice[] {
  // GGUF placement may use a different namespace from the global torch view.
  const pinnableBackend = data?.gpu?.gguf_gpu_ids_supported !== false;
  const diffusionBackend =
    data?.device_backend === "cuda" || data?.device_backend === "rocm";
  return (data?.gpu?.gguf_devices ?? data?.gpu?.devices ?? [])
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
      diffusionPinnable:
        diffusionBackend && d.index_kind === "physical",
    }));
}

/** Aggregate GPU info from /api/system; shares one module-level fetch across all GPU hooks. */
function useGpuInfoSource(source: "gpu" | "inference_gpu"): GpuInfo {
  const [gpu, setGpu] = useState<GpuInfo>(
    cachedSystem ? toGpuInfo(cachedSystem, source) : DEFAULT_GPU,
  );
  useEffect(() => {
    // No early return on cachedSystem: a consumer mounting as the cache fills
    // (between render and effect) would otherwise stay stuck at the default.
    let cancelled = false;
    let retryId: number | undefined;
    const update = (force = false, retryVulkan = false) => {
      fetchSystemOnce(force).then((d) => {
        if (cancelled) return;
        if (!d) {
          // Once an unavailable Vulkan backend starts polling, a transient API
          // failure must preserve the current state and continue the same loop.
          if (retryVulkan) {
            retryId = window.setTimeout(() => update(true, true), 3000);
          }
          return;
        }
        setGpu(toGpuInfo(d, source));
        const inferenceGpu = d.inference_gpu;
        if (
          source === "inference_gpu" &&
          inferenceGpu?.backend === "vulkan" &&
          !inferenceGpu.available
        ) {
          retryId = window.setTimeout(() => update(true, true), 3000);
        }
      });
    };
    update();
    return () => {
      cancelled = true;
      if (retryId !== undefined) window.clearTimeout(retryId);
    };
  }, [source]);
  return gpu;
}

/** Training-capable GPU info from the PyTorch/MLX hardware detector. */
export function useGpuInfo(): GpuInfo {
  return useGpuInfoSource("gpu");
}

/** GGUF inference GPU info, including a separately installed Vulkan backend. */
export function useInferenceGpuInfo(): GpuInfo {
  return useGpuInfoSource("inference_gpu");
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
    let lastSerialized: string | null = null;
    const sync = (data: SystemInfoResponse | null) => {
      if (cancelled) return;
      const next = toGpuDevices(data);
      // Every refresh builds a fresh array, so compare by value or a 3s Vulkan
      // retry loop would re-render this hook forever.
      const serialized = JSON.stringify(next);
      if (serialized === lastSerialized) return;
      lastSerialized = serialized;
      setDevices(next);
    };
    systemSubscribers.add(sync);
    fetchSystemOnce().then(sync);
    return () => {
      cancelled = true;
      systemSubscribers.delete(sync);
    };
  }, []);
  return devices;
}

/** Warm the shared system cache before validating persisted GPU IDs. */
export async function ensureGpuDeviceCache(): Promise<void> {
  await fetchSystemOnce();
}

/** Cached pinnable IDs, null before fetch, or [] when pinning is unavailable. */
export function cachedPinnableGpuIndices(
  forDiffusion = false,
): number[] | null {
  if (!cachedSystem) return null;
  const pinnable = toGpuDevices(cachedSystem).filter((d) =>
    forDiffusion ? d.diffusionPinnable : d.pinnable,
  );
  return pinnable.length > 1 ? pinnable.map((d) => d.index) : [];
}

/** Cached index namespace, undefined before fetch and null when unavailable. */
export function cachedPinnableGpuIndexKind(
  forDiffusion = false,
): GpuIndexKind | null | undefined {
  if (!cachedSystem) return undefined;
  const pinnable = toGpuDevices(cachedSystem).filter((d) =>
    forDiffusion ? d.diffusionPinnable : d.pinnable,
  );
  const kinds = new Set(pinnable.map((d) => d.indexKind).filter((k) => k));
  return pinnable.length > 1 && kinds.size === 1
    ? ([...kinds][0] as GpuIndexKind)
    : null;
}
