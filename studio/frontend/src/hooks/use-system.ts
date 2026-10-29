// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useMemo, useState } from "react";
import { disposableTimeoutSignal } from "../features/hub/lib/abort-signals";
import {
  isReadingStale,
  nextFailedPolls,
  settledFailureStatus,
  shouldRetrySystemDiscovery,
  type SystemInfoStatus,
  vulkanRetryDelayMs,
} from "./system-discovery";

export interface GpuDevice {
  index?: number;
  index_kind?: string;
  visible_ordinal?: number;
  name?: string;
  memory_total_gb?: number;
  vram_used_gb?: number;
  vram_free_gb?: number;
  vram_utilization_pct?: number | null;
  shared_memory?: boolean;
  /** A unified host pool (ROCm APU): a total, but not a VRAM ceiling. */
  unified_memory?: boolean;
  /** Host-backed part of the shared pool; the rest is reserved GPU memory. */
  shared_memory_host_backed_gb?: number | null;
  /** Apple MLX only: Metal's recommendedMaxWorkingSetSize. Absent elsewhere and on older backends. */
  metal_working_set_gb?: number | null;
  /** Live readings from the same probe as vram_used_gb; absent where the backend has none. */
  gpu_utilization_pct?: number | null;
  temperature_c?: number | null;
  power_draw_w?: number | null;
  power_limit_w?: number | null;
}

export interface SystemGpuInfo {
  available: boolean;
  backend?: string;
  /** Windows ROCm only, when no single device's usage could be attributed (#7452). */
  vram_used_gb_aggregate?: number | null;
  gguf_gpu_ids_supported?: boolean;
  backend_cuda_visible_devices?: string | null;
  parent_visible_gpu_ids?: number[];
  index_kind?: string;
  devices: GpuDevice[];
}

export { aggregateGpuMemoryTotalGb } from "./gpu-vram";

export interface SystemInfoResponse {
  memory_refreshed?: boolean;
  /** Client-side. Check before rendering a host verdict, or the placeholder reads as fact. */
  status: SystemInfoStatus;
  stale?: boolean;
  platform: string;
  python_version: string;
  device_backend: "cuda" | "rocm" | "cpu" | "mlx" | "xpu";
  /** Absent on older backends. */
  dense_quant_supported?: boolean;
  /** Best first. Absent on older backends, where readers default to []. */
  dense_quant_schemes?: string[];
  /** Absent on older backends, where readers treat it as off. */
  nvfp4_diffusion?: boolean;
  /** Prequantized LLM checkpoint formats ("fp8", "nvfp4") every card runs natively. Absent on older backends. */
  checkpoint_quant_formats?: string[];
  /** Whether group offload can stream torchao weights. Absent on older backends. */
  quantised_streaming?: boolean;
  /** GiB VRAM / GiB available RAM per lower-cased repo id. Absent on older backends. */
  diffusers_offload_tiers?: Record<
    string,
    Array<{
      gpu_gb: number;
      system_ram_gb: number;
      requires_quantised_streaming?: boolean;
    }>
  >;
  uptime_seconds: number | null;
  cpu: {
    logical_count: number;
    physical_count: number;
    usage_percent: number;
    frequency_mhz: number | null;
  };
  memory: {
    total_gb: number;
    available_gb: number;
    percent_used: number;
    process_used_mb: number;
  };
  disk: {
    total_gb: number;
    free_gb: number;
    percent_used: number;
  };
  /** null or absent when it is the system disk. */
  models_disk?: {
    total_gb: number;
    free_gb: number;
    percent_used: number;
  } | null;
  gpu: SystemGpuInfo;
  /** Differs from `gpu` when llama.cpp uses Vulkan. */
  inference_gpu?: SystemGpuInfo;
  ml_packages: {
    torch?: string;
    transformers?: string;
    mlx_lm?: string;
    mlx_vlm?: string;
  };
}

let cachedSystem: SystemInfoResponse | null = null;
let systemFetchPromise: Promise<SystemInfoResponse | null> | null = null;
const systemSubscribers = new Set<(data: SystemInfoResponse) => void>();
let vulkanRetrySubscribers = 0;
let vulkanRetryId: number | null = null;
let vulkanRetryAttempt = 0;
let vulkanRetryDeferred = false;
// A hung request must not hold systemFetchPromise, which every refreshMemory caller awaits.
const SYSTEM_FETCH_TIMEOUT_MS = 15_000;

const DEFAULT_SYSTEM: SystemInfoResponse = {
  status: "pending",
  platform: "Unknown",
  python_version: "Unknown",
  device_backend: "cpu",
  uptime_seconds: 0,
  cpu: { logical_count: 0, physical_count: 0, usage_percent: 0, frequency_mhz: null },
  memory: { total_gb: 0, available_gb: 0, percent_used: 0, process_used_mb: 0 },
  disk: { total_gb: 0, free_gb: 0, percent_used: 0 },
  gpu: { available: false, devices: [] },
  ml_packages: {}
};

export function getCachedSystemInfo(): SystemInfoResponse | null {
  return cachedSystem;
}

export function subscribeSystemInfo(
  subscriber: (data: SystemInfoResponse) => void,
  options: { retryUnavailableVulkan?: boolean } = {},
): () => void {
  systemSubscribers.add(subscriber);
  if (options.retryUnavailableVulkan) {
    vulkanRetrySubscribers += 1;
    scheduleVulkanRetry();
  }
  return () => {
    systemSubscribers.delete(subscriber);
    if (options.retryUnavailableVulkan) {
      vulkanRetrySubscribers = Math.max(0, vulkanRetrySubscribers - 1);
      if (vulkanRetrySubscribers === 0) {
        if (vulkanRetryId !== null) window.clearTimeout(vulkanRetryId);
        vulkanRetryId = null;
        if (vulkanRetryDeferred) cancelDeferredVulkanRetry();
      }
    }
  };
}

function cancelDeferredVulkanRetry(): void {
  vulkanRetryDeferred = false;
  document.removeEventListener("visibilitychange", retryVulkanWhenVisible);
}

function retryVulkanWhenVisible(): void {
  if (document.hidden) return;
  cancelDeferredVulkanRetry();
  void fetchSystemInfo({ force: true });
}

function scheduleVulkanRetry(): void {
  if (
    !shouldRetrySystemDiscovery(
      cachedSystem === null,
      cachedSystem?.inference_gpu,
      vulkanRetrySubscribers,
    )
  ) {
    // Cancel the pending cold-start retry once discovery succeeds.
    if (vulkanRetryId !== null) {
      window.clearTimeout(vulkanRetryId);
      vulkanRetryId = null;
    }
    vulkanRetryAttempt = 0;
    if (vulkanRetryDeferred) cancelDeferredVulkanRetry();
    return;
  }
  if (vulkanRetryId !== null || vulkanRetryDeferred) {
    return;
  }
  vulkanRetryId = window.setTimeout(
    () => {
      vulkanRetryId = null;
      if (document.hidden) {
        vulkanRetryDeferred = true;
        document.addEventListener("visibilitychange", retryVulkanWhenVisible);
        return;
      }
      void fetchSystemInfo({ force: true });
    },
    vulkanRetryDelayMs(vulkanRetryAttempt++),
  );
}

export async function fetchSystemInfo({
  force = false,
  refreshMemory = false,
}: { force?: boolean; refreshMemory?: boolean } = {}): Promise<SystemInfoResponse | null> {
  if (systemFetchPromise) {
    if (!refreshMemory) return systemFetchPromise;
    // An in-flight snapshot may predate the resident model.
    await systemFetchPromise;
    return fetchSystemInfo({ force, refreshMemory });
  }
  if (!force && !refreshMemory && cachedSystem) return cachedSystem;

  systemFetchPromise = (async () => {
    const timeout = disposableTimeoutSignal(SYSTEM_FETCH_TIMEOUT_MS);
    try {
      const res = await authFetch(
        refreshMemory ? "/api/system?refresh_memory=true" : "/api/system",
        { signal: timeout.signal },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      cachedSystem = { ...(data as SystemInfoResponse), status: "ready" };
      systemSubscribers.forEach((subscriber) => subscriber(cachedSystem!));
      return cachedSystem;
    } catch {
      return null;
    } finally {
      timeout.dispose();
      systemFetchPromise = null;
      scheduleVulkanRetry();
    }
  })();

  return systemFetchPromise;
}

interface UseSystemInfoOptions {
  pollMs?: number;
  enabled?: boolean;
}

export function useSystemInfo({
  pollMs,
  enabled = true,
}: UseSystemInfoOptions = {}): SystemInfoResponse {
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse>(cachedSystem ?? DEFAULT_SYSTEM);
  const [failedPolls, setFailedPolls] = useState(0);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let inFlight = false;
    let timeoutId: number | null = null;

    // A placeholder takes any published read; a real reading is left to this hook's poll.
    const unsubscribe = subscribeSystemInfo((info) => {
      if (cancelled) return;
      setSystemInfo((previous) => (previous.status === "ready" ? previous : info));
    });

    const update = (force: boolean) => {
      inFlight = true;
      void fetchSystemInfo({ force })
        .then((info) => {
          if (cancelled) return;
          setFailedPolls((previous) => nextFailedPolls(previous, info !== null));
          if (info) {
            setSystemInfo(info);
            return;
          }
          setSystemInfo((previous) => {
            const status = settledFailureStatus(previous.status);
            return status === previous.status
              ? previous
              : { ...DEFAULT_SYSTEM, status };
          });
        })
        .finally(() => {
          inFlight = false;
          if (cancelled || !pollMs) return;
          timeoutId = window.setTimeout(() => {
            timeoutId = null;
            if (!document.hidden) update(true);
          }, pollMs);
        });
    };

    const onVisibilityChange = () => {
      if (document.hidden || inFlight || cancelled) return;
      if (timeoutId !== null) window.clearTimeout(timeoutId);
      timeoutId = null;
      update(true);
    };

    update(Boolean(pollMs));
    if (pollMs) document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      cancelled = true;
      unsubscribe();
      if (timeoutId !== null) window.clearTimeout(timeoutId);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [enabled, pollMs]);

  const stale = enabled && isReadingStale(systemInfo.status, failedPolls, Boolean(pollMs));
  return useMemo(
    () => (stale ? { ...systemInfo, stale: true } : systemInfo),
    [stale, systemInfo],
  );
}
