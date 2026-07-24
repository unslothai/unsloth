// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import type { SystemInfoResponse } from "./use-system";

export interface GpuInfo {
  available: boolean;
  name: string;
  memoryTotalGb: number;
  /** VRAM budget for GGUF/llama-server workloads. On a Vulkan build this sums
   * the devices llama-server actually uses (gguf_devices), which can include
   * cards the torch backend can't see (e.g. a pre-ROCm AMD card); otherwise
   * identical to memoryTotalGb. GGUF fit labels must use this; torch-based
   * (training / safetensors) estimates must stay on memoryTotalGb. */
  ggufMemoryTotalGb: number;
  cpuCore: number;
  cpuThread: number;
  systemRamAvailableGb: number;
  systemRamTotalGb: number
}

export interface SystemGpuDevice {
  index: number;
  name: string;
  memoryTotalGb: number;
  /** Free VRAM at fetch time. Degrades to the total when the utilization
   * probe had no usage data; 0 only when the total is unknown too. */
  memoryFreeGb: number;
  /** True when `index` is safe to pin via gpu_ids: a stable physical/PCI id,
   *  or a ggml Vulkan ordinal (the space /load pins with --device Vulkan<i>).
   *  False for ordinals into a parent CUDA_VISIBLE_DEVICES mask, which the
   *  backend can't map back, so the picker must not offer them. */
  physicalIndex: boolean;
}

const DEFAULT_GPU: GpuInfo = {
  available: false,
  name: "Unknown",
  memoryTotalGb: 0,
  ggufMemoryTotalGb: 0,
  cpuCore: 0,
  cpuThread: 0,
  systemRamAvailableGb: 0,
  systemRamTotalGb: 0
};

// One module-level cache so every GPU hook shares a single /api/system fetch.
let cachedSystem: SystemInfoResponse | null = null;
let systemPromise: Promise<SystemInfoResponse | null> | null = null;

// Persisted gpu_ids picks are bare numbers, so they only mean "these cards"
// within the index space they were saved in: physical CUDA/ROCm ids, or ggml
// Vulkan ordinals on a Vulkan build. Swapping the llama.cpp backend flips that
// space while keeping many numbers valid (Vulkan ordinal 1 exists but may be a
// different card than physical id 1). Each saved pick is stamped with the kind
// it was made under (PerModelConfig.selectedGpuIdsIndexKind, and the runtime
// store's selectedGpuIdsKind); reconcilePersistedGpuIds drops a pick whose stamp
// no longer matches the kind the backend reports here.
export type GpuIndexKind = "vulkan" | "physical";

function systemGpuIndexKind(
  data: SystemInfoResponse | null,
): GpuIndexKind | null {
  if (!data) return null;
  return (data.gpu?.gguf_devices ?? []).length ? "vulkan" : "physical";
}

/** The index space the backend currently reports gpu_ids in, or null when the
 * /api/system cache has not populated yet (so callers cannot decide and must
 * defer the index-space judgement to a later warm reconcile). */
export function currentGpuIndexKind(): GpuIndexKind | null {
  return systemGpuIndexKind(cachedSystem);
}

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
  // GGUF budget: what llama-server can use. Skips iGPUs (their "VRAM" is the
  // shared system RAM already reported above). When the backend reports a
  // separate llama-server inventory it is authoritative EVEN at zero discrete
  // VRAM (e.g. llama-server masked to an iGPU while torch still sees a dGPU
  // that llama-server won't use); only its absence falls back to the torch
  // total.
  const ggufDevices = gpuData?.gguf_devices ?? [];
  const ggufDeviceTotalGb = ggufDevices.reduce(
    (sum, d) => sum + (d.is_igpu ? 0 : (d.memory_total_gb ?? 0)),
    0,
  );
  if (!gpuData?.available || !devices.length) {
    // Torch sees no GPU (training stays CPU-bound / unavailable), but a Vulkan
    // llama.cpp build may still drive GPUs for GGUF: surface that budget alone.
    return { ...DEFAULT_GPU, ...base, ggufMemoryTotalGb: ggufDeviceTotalGb };
  }
  const memoryTotalGb = devices.reduce(
    (sum, d) => sum + (d.memory_total_gb ?? 0),
    0,
  );
  return {
    ...base,
    available: true,
    name: devices[0]?.name ?? "Unknown",
    memoryTotalGb,
    ggufMemoryTotalGb: ggufDevices.length ? ggufDeviceTotalGb : memoryTotalGb,
  };
}

function toGpuDevices(data: SystemInfoResponse | null): SystemGpuDevice[] {
  // Unpinnable configurations must hide every pick surface: XPU indices are
  // torch-xpu ordinals no applicator speaks -- /load and /validate 400 picks,
  // so the backend reports gpu.gguf_gpu_ids_supported and every gate keyed on
  // physicalIndex (picker, persisted-pick reconcile) follows it. The device
  // flavor lives on the TOP-LEVEL device_backend field; absent support info
  // defaults to pinnable (older backend).
  const pinnableBackend =
    data?.device_backend !== "xpu" &&
    data?.gpu?.gguf_gpu_ids_supported !== false;
  // These devices exist to drive GGUF loads, so when the backend reports the
  // llama-server (Vulkan) inventory, that list is authoritative: it can see
  // cards torch can't, its indices are the ggml ordinals /load pins with
  // --device Vulkan<i>, and gguf_gpu_ids_supported says picks are accepted.
  const ggufDevices = data?.gpu?.gguf_devices ?? [];
  if (ggufDevices.length) {
    return ggufDevices
      .filter((d) => typeof d.index === "number")
      .map((d) => ({
        index: d.index as number,
        name: d.name ?? `GPU ${d.index}`,
        memoryTotalGb: d.memory_total_gb ?? 0,
        memoryFreeGb: d.vram_free_gb ?? 0,
        physicalIndex: pinnableBackend && d.index_kind === "vulkan",
      }));
  }
  return (data?.gpu?.devices ?? [])
    .filter((d) => typeof d.index === "number")
    .map((d) => ({
      index: d.index as number,
      name: d.name ?? `GPU ${d.index}`,
      memoryTotalGb: d.memory_total_gb ?? 0,
      memoryFreeGb: d.vram_free_gb ?? 0,
      physicalIndex: pinnableBackend && d.index_kind === "physical",
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

/** Reactive currentGpuIndexKind(): re-renders when /api/system loads, so a
 *  component reading a pick's index space (e.g. the active model's) updates
 *  once the cache warms instead of being stuck at the cold-cache null. */
export function useCurrentGpuIndexKind(): GpuIndexKind | null {
  const [kind, setKind] = useState<GpuIndexKind | null>(() =>
    systemGpuIndexKind(cachedSystem),
  );
  useEffect(() => {
    let cancelled = false;
    fetchSystemOnce().then((d) => {
      if (!cancelled) setKind(systemGpuIndexKind(d));
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return kind;
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
  const physical = toGpuDevices(cachedSystem).filter((d) => d.physicalIndex);
  // Mirrors the sheet's showGpuPicker gate: only a 2+ physical-GPU host can pin.
  return physical.length > 1 ? physical.map((d) => d.index) : [];
}
