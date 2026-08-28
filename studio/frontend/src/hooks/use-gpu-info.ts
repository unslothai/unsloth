// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState } from "react";
import {
  type GpuIndexKind,
  type PinnableGpuContext,
  type ReconciledGpuSelection,
  type SystemGpuDevice,
  pickLoadDevice,
  pinnableGpuContext,
  reconcileGpuSelection,
  resolveGpuSelectionContext,
} from "./gpu-selection";
import {
  gpuMemoryTotalsGb,
  gpuSharedHostMemoryGb,
  systemRamAvailableOutsideSharedPoolGb,
} from "./gpu-vram";
import {
  type SystemInfoResponse,
  fetchSystemInfo,
  getCachedSystemInfo,
  subscribeSystemInfo,
} from "./use-system";

export {
  pinnableGpuContext,
  reconcileGpuSelection,
  type GpuIndexKind,
  type PinnableGpuContext,
  type ReconciledGpuSelection,
  type SystemGpuDevice,
} from "./gpu-selection";

export interface GpuInfo {
  available: boolean;
  budgetKnown: boolean;
  /** true when the visible GPUs use only the host memory pool. */
  sharedMemory: boolean;
  /** True when any device's budget is a unified host pool (a ROCm APU), which is
   *  not a VRAM ceiling a fit verdict can be measured against. */
  unifiedMemory: boolean;
  /** The backend torch resolved: cuda, rocm, xpu, mlx, cpu. Carried on every path, including the
   * GPU-less one, because "which runtimes can this host place" is exactly the question a host
   * with no usable GPU has to answer. Empty until system info arrives. */
  backend: string;
  name: string;
  memoryTotalGb: number;
  memorySharedGb: number;
  /** The same aggregate with shared-memory devices left out: the VRAM that is a pool
   *  BESIDE system RAM rather than a capped view of it. */
  dedicatedMemoryTotalGb: number;
  /** Largest single device's VRAM. Image/video loads live on ONE device (no tensor split), so their fit math must use a single device, not the multi-GPU sum. */
  maxDeviceMemoryGb: number;
  /** VRAM of the device an image/video load actually lands on: the lowest visible ordinal, since resolve_diffusion_device_target() returns a bare "cuda" and torch places on the current device. On a heterogeneous host this is NOT maxDeviceMemoryGb, and sizing a pick against the larger card would recommend a checkpoint that OOMs the smaller one. */
  loadDeviceMemoryGb: number;
  /** true when the image/video load device uses the host memory pool. */
  loadDeviceSharedMemory: boolean;
  cpuCore: number;
  cpuThread: number;
  /** host RAM free after removing the host-backed shared GPU pool. */
  systemRamAvailableGb: number;
  /** raw host RAM free as the probe reported it. */
  systemRamAvailableHostGb: number;
  systemRamTotalGb: number;
  /** Free space on the cache volume. Not a GPU fact, carried here because it is the
   *  floor under every other budget: a GGUF too big to land cannot be paged from a
   *  file that was never written. */
  diskFreeGb: number;
}

const DEFAULT_GPU: GpuInfo = {
  available: false,
  budgetKnown: false,
  sharedMemory: false,
  unifiedMemory: false,
  backend: "",
  name: "Unknown",
  memoryTotalGb: 0,
  memorySharedGb: 0,
  dedicatedMemoryTotalGb: 0,
  maxDeviceMemoryGb: 0,
  loadDeviceMemoryGb: 0,
  loadDeviceSharedMemory: false,
  cpuCore: 0,
  cpuThread: 0,
  systemRamAvailableGb: 0,
  systemRamAvailableHostGb: 0,
  systemRamTotalGb: 0,
  diskFreeGb: 0,
};

function toGpuInfo(
  data: SystemInfoResponse | null,
  source: "gpu" | "inference_gpu" = "gpu",
): GpuInfo {
  // CPU/RAM exist even on GPU-less hosts (e.g. Mac), so populate them on every
  // path: unified-memory math still needs a RAM budget to work with.
  const base = {
    backend: data?.device_backend ?? "",
    cpuCore: data?.cpu?.physical_count ?? 0,
    cpuThread: data?.cpu?.logical_count ?? 0,
    systemRamAvailableGb: data?.memory?.available_gb ?? 0,
    systemRamAvailableHostGb: data?.memory?.available_gb ?? 0,
    systemRamTotalGb: data?.memory?.total_gb ?? 0,
    // The cache volume when the backend reports it (downloads land there, and it
    // can be a different disk than the root), else the root figure.
    diskFreeGb: data?.disk?.cache_free_gb ?? data?.disk?.free_gb ?? 0,
  };
  const gpuData =
    source === "inference_gpu" ? (data?.inference_gpu ?? data?.gpu) : data?.gpu;
  const devices = gpuData?.devices ?? [];
  if (!gpuData?.available || !devices.length) {
    return { ...DEFAULT_GPU, ...base, budgetKnown: data !== null };
  }
  const memoryTotals = gpuMemoryTotalsGb(devices);
  const loadDevice = pickLoadDevice(devices);
  return {
    ...base,
    systemRamAvailableGb: systemRamAvailableOutsideSharedPoolGb(
      base.systemRamAvailableGb,
      gpuSharedHostMemoryGb(devices),
    ),
    sharedMemory: memoryTotals.shared > 0 && memoryTotals.dedicated === 0,
    // Additive, and deliberately some() where sharedMemory above is "no dedicated
    // pool at all": one unified part makes the aggregate total partly host RAM,
    // which is already enough to stop it being a VRAM ceiling a fit verdict can
    // be measured against.
    unifiedMemory: devices.some((device) => device.unified_memory === true),
    available: true,
    budgetKnown: true,
    name: devices[0]?.name ?? "Unknown",
    memoryTotalGb: memoryTotals.total,
    dedicatedMemoryTotalGb: memoryTotals.dedicated,
    memorySharedGb: memoryTotals.shared,
    maxDeviceMemoryGb: devices.reduce(
      (max, d) => Math.max(max, d.memory_total_gb ?? 0),
      0,
    ),
    // Lowest visible ordinal = torch's current device = where the pipeline lands.
    loadDeviceMemoryGb: loadDevice?.memory_total_gb ?? 0,
    loadDeviceSharedMemory: loadDevice?.shared_memory === true,
  };
}

function toGpuDevices(
  data: SystemInfoResponse | null,
  // Diffusion runs on torch, not llama-server, so it reads the torch inventory even where the
  // inference backend is Vulkan: those are separate runtimes and a Vulkan chat build says nothing
  // about the CUDA / ROCm devices an image or video load can be pinned to.
  forDiffusion = false,
): SystemGpuDevice[] {
  // GGUF loads run through llama-server, so on a Vulkan build the pickable set
  // is the inference inventory, not the torch view: it can see cards torch
  // cannot, and its indices are the ggml ordinals `--device Vulkan<i>` pins.
  // The XPU ban does not apply there, it is about torch-xpu ordinals that no
  // applicator speaks; a Vulkan pick does not use them.
  const inference = data?.inference_gpu;
  if (!forDiffusion && inference?.backend === "vulkan") {
    // The installed inference backend is confirmed Vulkan, so even an empty
    // device list (probe still cold, or transiently failed) must NOT fall
    // through to the torch/CUDA inventory below: those physical IDs are
    // meaningless to a Vulkan llama-server, and the backend rejects every
    // explicit diffusion pin outright while is_vulkan_build is true. Report no
    // pinnable/diffusionPinnable devices until the probe succeeds.
    if (!(inference.devices ?? []).length) return [];
    const picksAccepted = inference.gguf_gpu_ids_supported !== false;
    return (inference.devices ?? [])
      .filter((d) => typeof d.index === "number")
      .map((d) => ({
        index: d.index as number,
        indexKind: d.index_kind === "vulkan" ? ("vulkan" as const) : null,
        name: d.name ?? `GPU ${d.index}`,
        memoryTotalGb: d.memory_total_gb ?? 0,
        memoryFreeGb: d.vram_free_gb ?? 0,
        sharedMemory: d.shared_memory === true,
        sharedMemoryHostBackedGb: d.shared_memory_host_backed_gb,
        unifiedMemory: d.unified_memory === true,
        pinnable: picksAccepted && d.index_kind === "vulkan",
        // The DiffusionGemma runner is torch-side and never speaks ggml
        // ordinals, so a Vulkan pick is not usable there.
        diffusionPinnable: false,
      }));
  }
  // Otherwise the torch view is the pickable set. Unpinnable configurations
  // must hide every pick surface: the backend reports gguf_gpu_ids_supported,
  // and absent support info defaults to pinnable (older backend).
  const pinnableBackend = data?.gpu?.gguf_gpu_ids_supported !== false;
  // ROCm reuses torch.cuda.* and the same physical-ID path, so the runner takes
  // its indices too; only the reported label differs (_backend_label swaps it).
  const diffusionBackend =
    data?.device_backend === "cuda" || data?.device_backend === "rocm";
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
      sharedMemory: d.shared_memory === true,
      sharedMemoryHostBackedGb: d.shared_memory_host_backed_gb,
      unifiedMemory: d.unified_memory === true,
      // The XPU ban is about torch-xpu ordinals no applicator speaks, so /load
      // and /validate 400 them. A Vulkan ordinal is not one of those, so it
      // stays pickable even when this list arrives from an XPU host.
      pinnable:
        pinnableBackend &&
        (d.index_kind === "vulkan" ||
          (data?.device_backend !== "xpu" && d.index_kind === "physical")),
      diffusionPinnable: diffusionBackend && d.index_kind === "physical",
    }));
}

/** Aggregate GPU info from /api/system; shares one module-level fetch across all GPU hooks. */
function useGpuInfoSource(source: "gpu" | "inference_gpu"): GpuInfo {
  const cachedSystem = getCachedSystemInfo();
  const [gpu, setGpu] = useState<GpuInfo>(
    cachedSystem ? toGpuInfo(cachedSystem, source) : DEFAULT_GPU,
  );
  useEffect(() => {
    // No early return on cachedSystem: a consumer mounting as the cache fills
    // (between render and effect) would otherwise stay stuck at the default.
    let cancelled = false;
    const sync = (data: SystemInfoResponse) => {
      if (cancelled) return;
      const next = toGpuInfo(data, source);
      setGpu((current) =>
        JSON.stringify(current) === JSON.stringify(next) ? current : next,
      );
    };
    const update = () => {
      fetchSystemInfo().then((d) => {
        if (cancelled) return;
        if (!d) return;
        // A cache hit does not publish a new snapshot. Sync it here so a
        // consumer mounting while the initial request finishes cannot miss it.
        sync(d);
      });
    };
    const unsubscribe = subscribeSystemInfo(sync, {
      retryUnavailableVulkan: source === "inference_gpu",
    });
    update();
    return () => {
      cancelled = true;
      unsubscribe();
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
export function useGpuDevices(forDiffusion = false): SystemGpuDevice[] {
  const cachedSystem = getCachedSystemInfo();
  const [devices, setDevices] = useState<SystemGpuDevice[]>(
    cachedSystem ? toGpuDevices(cachedSystem, forDiffusion) : [],
  );
  useEffect(() => {
    // No early return on cachedSystem: a consumer mounting as the cache fills
    // (between render and effect) would otherwise stay stuck at the default.
    let cancelled = false;
    let lastSerialized: string | null = null;
    const sync = (data: SystemInfoResponse | null) => {
      if (cancelled) return;
      const next = toGpuDevices(data, forDiffusion);
      // Every refresh builds a fresh array, so compare by value or a 3s Vulkan
      // retry loop would re-render this hook forever.
      const serialized = JSON.stringify(next);
      if (serialized === lastSerialized) return;
      lastSerialized = serialized;
      setDevices(next);
    };
    const unsubscribe = subscribeSystemInfo(sync, {
      retryUnavailableVulkan: true,
    });
    fetchSystemInfo().then(sync);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [forDiffusion]);
  return devices;
}

/**
 * Cards an image or video load may be pinned to, or empty when there is nothing to choose
 * between. Neither engine shards a diffusion checkpoint, so this drives a single-choice control
 * rather than the chat picker's candidate pool.
 */
export function useDiffusionGpuChoices(): SystemGpuDevice[] {
  const devices = useGpuDevices(true);
  // Memoized on the device list, which only changes when the inventory does. pinnableGpuContext
  // builds a fresh filtered array per call, so an unmemoized return changed identity on every
  // render of the page holding it: it feeds the load-advanced snapshot, that feeds the download
  // footprint resolver, and the GGUF picker re-runs its effect whenever the resolver changes --
  // clearing the sizes it had and re-POSTing a download plan per variant on every status poll.
  return useMemo(() => {
    const context = pinnableGpuContext(devices, true);
    return (context.ids?.length ?? 0) > 1 ? (context.devices ?? []) : [];
  }, [devices]);
}

/** Whether device discovery is settled enough to rewrite remembered UI state. */
export function gpuDeviceCacheReady(): boolean {
  const cachedSystem = getCachedSystemInfo();
  if (cachedSystem === null) {
    return false;
  }
  const inferenceGpu = cachedSystem.inference_gpu;
  return !(inferenceGpu?.backend === "vulkan" && !inferenceGpu.available);
}

/** Warm the shared system cache before validating persisted GPU IDs. */
export async function ensureGpuDeviceCache(): Promise<void> {
  await fetchSystemInfo();
}

/** Cached pinnable IDs, null before fetch, or [] when pinning is unavailable. */
export function cachedPinnableGpuIndices(
  forDiffusion = false,
): number[] | null {
  return cachedPinnableGpuContext(forDiffusion).ids;
}

/** Cached index namespace, undefined before fetch and null when unavailable. */
export function cachedPinnableGpuIndexKind(
  forDiffusion = false,
): GpuIndexKind | null | undefined {
  return cachedPinnableGpuContext(forDiffusion).indexKind;
}

/**
 * Cached namespace and membership are separate: an unavailable Vulkan probe
 * leaves membership unknown while the Vulkan namespace remains authoritative.
 */
export function cachedPinnableGpuContext(
  forDiffusion = false,
  devices?: SystemGpuDevice[],
): PinnableGpuContext {
  const cachedSystem = getCachedSystemInfo();
  const unavailableVulkan =
    cachedSystem?.inference_gpu?.backend === "vulkan" &&
    !cachedSystem.inference_gpu.available;
  return resolveGpuSelectionContext(
    cachedSystem ? (devices ?? toGpuDevices(cachedSystem)) : null,
    forDiffusion,
    unavailableVulkan ? "vulkan" : undefined,
  );
}

export function reconcileCachedGpuSelection(
  ids: number[] | null,
  savedIndexKind?: GpuIndexKind | null,
  forDiffusion = false,
): ReconciledGpuSelection {
  const context = cachedPinnableGpuContext(forDiffusion);
  return reconcileGpuSelection(
    ids,
    savedIndexKind,
    context.indexKind,
    context.ids,
  );
}
