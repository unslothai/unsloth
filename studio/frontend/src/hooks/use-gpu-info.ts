// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState } from "react";
import { normalizeDenseQuantSchemes } from "@/lib/dense-quant-schemes";
import {
  type ReportedOffloadFitTier,
  normalizeReportedOffloadFitTiers,
} from "@/lib/offload-fit-tiers";
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
  sharesHostMemory,
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
  sharedMemory: boolean;
  /** A unified host pool (ROCm APU) is not a VRAM ceiling for fit verdicts. */
  unifiedMemory: boolean;
  /** Carried on the GPU-less path too. Empty until system info arrives. */
  backend: string;
  denseQuantSupported: boolean;
  /** Best first. Empty until system info arrives, and on older backends. */
  denseQuantSchemes: readonly string[];
  nvfp4Diffusion: boolean;
  quantisedStreaming?: boolean;
  extraOffloadFitTiers?: Readonly<
    Record<string, readonly ReportedOffloadFitTier[]>
  >;
  name: string;
  memoryTotalGb: number;
  memorySharedGb: number;
  /** Without shared-memory devices: VRAM beside system RAM, not a view of it. */
  dedicatedMemoryTotalGb: number;
  /** Image/video loads live on ONE device, so their fit math must not use the multi-GPU sum. */
  maxDeviceMemoryGb: number;
  /** Lowest visible ordinal, where a bare "cuda" diffusion load lands; NOT maxDeviceMemoryGb on a
   * heterogeneous host. */
  loadDeviceMemoryGb: number;
  loadDeviceSharedMemory: boolean;
  /** Includes Linux ROCm APUs, which report only unified_memory; offload frees nothing on either. */
  loadDeviceSharesHostMemory: boolean;
  /** For the loader's per-card VRAM reserve. */
  deviceCount: number;
  cpuCore: number;
  cpuThread: number;
  /** Host RAM free minus the host-backed shared GPU pool. */
  systemRamAvailableGb: number;
  systemRamAvailableHostGb: number;
  /** Whether host available memory was reported, including a real zero. */
  systemRamAvailableKnown?: boolean;
  systemRamTotalGb: number;
}

const DEFAULT_GPU: GpuInfo = {
  available: false,
  budgetKnown: false,
  sharedMemory: false,
  unifiedMemory: false,
  backend: "",
  denseQuantSupported: false,
  denseQuantSchemes: [],
  nvfp4Diffusion: false,
  quantisedStreaming: false,
  name: "Unknown",
  memoryTotalGb: 0,
  memorySharedGb: 0,
  dedicatedMemoryTotalGb: 0,
  maxDeviceMemoryGb: 0,
  loadDeviceMemoryGb: 0,
  loadDeviceSharedMemory: false,
  loadDeviceSharesHostMemory: false,
  deviceCount: 0,
  cpuCore: 0,
  cpuThread: 0,
  systemRamAvailableGb: 0,
  systemRamAvailableHostGb: 0,
  systemRamAvailableKnown: false,
  systemRamTotalGb: 0,
};

function toGpuInfo(
  data: SystemInfoResponse | null,
  source: "gpu" | "inference_gpu" = "gpu",
): GpuInfo {
  // CPU/RAM exist on GPU-less hosts too, and unified-memory math needs a RAM budget.
  const base = {
    backend: data?.device_backend ?? "",
    denseQuantSupported: data?.dense_quant_supported === true,
    denseQuantSchemes: normalizeDenseQuantSchemes(data?.dense_quant_schemes),
    nvfp4Diffusion: data?.nvfp4_diffusion === true,
    quantisedStreaming: data?.quantised_streaming === true,
    extraOffloadFitTiers: normalizeReportedOffloadFitTiers(
      data?.diffusers_offload_tiers,
    ),
    cpuCore: data?.cpu?.physical_count ?? 0,
    cpuThread: data?.cpu?.logical_count ?? 0,
    systemRamAvailableGb: data?.memory?.available_gb ?? 0,
    systemRamAvailableHostGb: data?.memory?.available_gb ?? 0,
    systemRamAvailableKnown:
      Number.isFinite(data?.memory?.available_gb) &&
      (data?.memory?.available_gb as number) >= 0,
    systemRamTotalGb: data?.memory?.total_gb ?? 0,
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
    // Raw: folding first collapsed multi-socket unified pools into one.
    systemRamAvailableGb: systemRamAvailableOutsideSharedPoolGb(
      base.systemRamAvailableGb,
      gpuSharedHostMemoryGb(devices),
    ),
    sharedMemory: memoryTotals.shared > 0 && memoryTotals.dedicated === 0,
    // some(), unlike sharedMemory: one unified part already stops the total being a VRAM ceiling.
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
    loadDeviceSharesHostMemory: sharesHostMemory({
      sharedMemory: loadDevice?.shared_memory === true,
      unifiedMemory: loadDevice?.unified_memory === true,
    }),
    deviceCount: devices.length,
  };
}

function toGpuDevices(
  data: SystemInfoResponse | null,
  // Diffusion runs on torch, so it reads the torch inventory even on a Vulkan inference build.
  forDiffusion = false,
): SystemGpuDevice[] {
  // GGUF runs via llama-server, so on Vulkan the pickable set is its inventory and ggml ordinals.
  const inference = data?.inference_gpu;
  if (!forDiffusion && inference?.backend === "vulkan") {
    // Confirmed Vulkan: never fall through to torch IDs, which the backend rejects for Vulkan builds.
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
        memoryFreeKnown:
          Number.isFinite(d.vram_free_gb) && (d.vram_free_gb as number) >= 0,
        sharedMemory: d.shared_memory === true,
        sharedMemoryHostBackedGb: d.shared_memory_host_backed_gb,
        unifiedMemory: d.unified_memory === true,
        pinnable: picksAccepted && d.index_kind === "vulkan",
        // The DiffusionGemma runner never speaks ggml ordinals.
        diffusionPinnable: false,
      }));
  }
  // Absent gguf_gpu_ids_supported (older backend) defaults to pinnable.
  const pinnableBackend = data?.gpu?.gguf_gpu_ids_supported !== false;
  // ROCm reuses torch.cuda.* and physical IDs; only the label differs.
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
      memoryFreeKnown:
        Number.isFinite(d.vram_free_gb) && (d.vram_free_gb as number) >= 0,
      sharedMemory: d.shared_memory === true,
      sharedMemoryHostBackedGb: d.shared_memory_host_backed_gb,
      unifiedMemory: d.unified_memory === true,
      // The XPU ban covers torch-xpu ordinals, not Vulkan ones.
      pinnable:
        pinnableBackend &&
        (d.index_kind === "vulkan" ||
          (data?.device_backend !== "xpu" && d.index_kind === "physical")),
      diffusionPinnable: diffusionBackend && d.index_kind === "physical",
    }));
}

/** Keep the previous array when unchanged: polls re-probe memory, and consumers memoise picker
 * options on this list. */
export function withStableSchemes(current: GpuInfo, next: GpuInfo): GpuInfo {
  const held = current.denseQuantSchemes;
  const fresh = next.denseQuantSchemes;
  if (held === fresh) return next;
  if (
    held.length === fresh.length &&
    held.every((scheme, index) => scheme === fresh[index])
  ) {
    return { ...next, denseQuantSchemes: held };
  }
  return next;
}

/** Shares one module-level fetch across all GPU hooks. */
function useGpuInfoSource(source: "gpu" | "inference_gpu"): GpuInfo {
  const cachedSystem = getCachedSystemInfo();
  const [gpu, setGpu] = useState<GpuInfo>(
    cachedSystem ? toGpuInfo(cachedSystem, source) : DEFAULT_GPU,
  );
  useEffect(() => {
    // No early return on cachedSystem: a consumer mounting as the cache fills would stay default.
    let cancelled = false;
    const sync = (data: SystemInfoResponse) => {
      if (cancelled) return;
      const next = toGpuInfo(data, source);
      setGpu((current) =>
        JSON.stringify(current) === JSON.stringify(next)
          ? current
          : withStableSchemes(current, next),
      );
    };
    const update = () => {
      fetchSystemInfo().then((d) => {
        if (cancelled) return;
        if (!d) return;
        // A cache hit publishes no snapshot, so sync here.
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

export function useGpuInfo(): GpuInfo {
  return useGpuInfoSource("gpu");
}

/** Includes a separately installed Vulkan backend. */
export function useInferenceGpuInfo(): GpuInfo {
  return useGpuInfoSource("inference_gpu");
}

export function isEngineGpuDevice(device: SystemGpuDevice): boolean {
  return device.indexKind === "physical" && /nvidia/i.test(device.name);
}

/** Where the backend puts an optional engine given no GPUs: the first one Studio sees. */
export function defaultEngineGpuIds(): number[] {
  const first = toGpuDevices(getCachedSystemInfo()).find(isEngineGpuDevice);
  return first ? [first.index] : [0];
}

export function useGpuDevices(forDiffusion = false): SystemGpuDevice[] {
  const cachedSystem = getCachedSystemInfo();
  const [devices, setDevices] = useState<SystemGpuDevice[]>(
    cachedSystem ? toGpuDevices(cachedSystem, forDiffusion) : [],
  );
  useEffect(() => {
    // No early return on cachedSystem: a consumer mounting as the cache fills would stay default.
    let cancelled = false;
    let lastSerialized: string | null = null;
    const sync = (data: SystemInfoResponse | null) => {
      if (cancelled) return;
      const next = toGpuDevices(data, forDiffusion);
      // Compare by value, or the 3s Vulkan retry would re-render forever.
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

/** Empty when there is nothing to choose; diffusion checkpoints are never sharded. */
export function useDiffusionGpuChoices(): SystemGpuDevice[] {
  const devices = useGpuDevices(true);
  // Memoized: a fresh array per render cascaded into the GGUF picker re-POSTing download plans
  // on every status poll.
  return useMemo(() => {
    const context = pinnableGpuContext(devices, true);
    return (context.ids?.length ?? 0) > 1 ? (context.devices ?? []) : [];
  }, [devices]);
}

export function gpuDeviceCacheReady(): boolean {
  const cachedSystem = getCachedSystemInfo();
  if (cachedSystem === null) {
    return false;
  }
  const inferenceGpu = cachedSystem.inference_gpu;
  return !(inferenceGpu?.backend === "vulkan" && !inferenceGpu.available);
}

export async function ensureGpuDeviceCache(): Promise<void> {
  await fetchSystemInfo();
}

/** null before fetch, [] when pinning is unavailable. */
export function cachedPinnableGpuIndices(
  forDiffusion = false,
): number[] | null {
  return cachedPinnableGpuContext(forDiffusion).ids;
}

/** undefined before fetch, null when unavailable. */
export function cachedPinnableGpuIndexKind(
  forDiffusion = false,
): GpuIndexKind | null | undefined {
  return cachedPinnableGpuContext(forDiffusion).indexKind;
}

/** An unavailable Vulkan probe leaves membership unknown while the namespace stays authoritative. */
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
