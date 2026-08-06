


import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import { shouldRetrySystemDiscovery } from "./system-discovery";

export interface GpuDevice {
  index?: number;
  index_kind?: string;
  visible_ordinal?: number;
  name?: string;
  memory_total_gb?: number;
  vram_used_gb?: number;
  vram_free_gb?: number;
  vram_utilization_pct?: number | null;
  /** True when the reported GPU budget comes from shared system memory. */
  shared_memory?: boolean;
}

export interface SystemGpuInfo {
  available: boolean;
  backend?: string;
  /** Whether GGUF loads accept explicit gpu_ids in the device records'
   * declared index space. */
  gguf_gpu_ids_supported?: boolean;
  backend_cuda_visible_devices?: string | null;
  parent_visible_gpu_ids?: number[];
  index_kind?: string;
  devices: GpuDevice[];
}

/** Sum dedicated VRAM while counting a shared host-memory pool only once. */
export function aggregateGpuMemoryTotalGb(devices: GpuDevice[]): number {
  const dedicated = devices
    .filter((device) => !device.shared_memory)
    .reduce((sum, device) => sum + (device.memory_total_gb ?? 0), 0);
  const shared = Math.max(
    0,
    ...devices
      .filter((device) => device.shared_memory)
      .map((device) => device.memory_total_gb ?? 0),
  );
  return dedicated + shared;
}

export interface SystemInfoResponse {
  platform: string;
  python_version: string;
  device_backend: "cuda" | "rocm" | "cpu" | "mlx" | "xpu";
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
  gpu: SystemGpuInfo;
  /** Devices available to GGUF inference; differs when llama.cpp uses Vulkan. */
  inference_gpu?: SystemGpuInfo;
  ml_packages: {
    torch?: string;
    transformers?: string;
  };
}

let cachedSystem: SystemInfoResponse | null = null;
let systemFetchPromise: Promise<SystemInfoResponse | null> | null = null;
const systemSubscribers = new Set<(data: SystemInfoResponse) => void>();
let vulkanRetrySubscribers = 0;
let vulkanRetryId: number | null = null;

const DEFAULT_SYSTEM: SystemInfoResponse = {
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
      if (vulkanRetrySubscribers === 0 && vulkanRetryId !== null) {
        window.clearTimeout(vulkanRetryId);
        vulkanRetryId = null;
      }
    }
  };
}

function scheduleVulkanRetry(): void {
  if (
    !shouldRetrySystemDiscovery(
      cachedSystem === null,
      cachedSystem?.inference_gpu,
      vulkanRetrySubscribers,
    )
  ) {
    // A cold subscription schedules before its first request settles. Cancel
    // that pending retry as soon as discovery succeeds with a usable inventory
    // or a non-Vulkan backend.
    if (vulkanRetryId !== null) {
      window.clearTimeout(vulkanRetryId);
      vulkanRetryId = null;
    }
    return;
  }
  if (vulkanRetryId !== null) {
    return;
  }
  vulkanRetryId = window.setTimeout(() => {
    vulkanRetryId = null;
    void fetchSystemInfo({ force: true });
  }, 3000);
}

export async function fetchSystemInfo({
  force = false,
}: { force?: boolean } = {}): Promise<SystemInfoResponse | null> {
  if (systemFetchPromise) return systemFetchPromise;
  if (!force && cachedSystem) return cachedSystem;

  systemFetchPromise = (async () => {
    try {
      const res = await authFetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      cachedSystem = data as SystemInfoResponse;
      systemSubscribers.forEach((subscriber) => subscriber(cachedSystem!));
      return cachedSystem;
    } catch {
      return null;
    } finally {
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

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let timeoutId: number | null = null;

    const update = (force: boolean) => {
      void fetchSystemInfo({ force })
        .then((info) => {
          if (!cancelled && info) setSystemInfo(info);
        })
        .finally(() => {
          if (cancelled || !pollMs) return;
          timeoutId = window.setTimeout(() => update(true), pollMs);
        });
    };

    update(Boolean(pollMs));
    return () => {
      cancelled = true;
      if (timeoutId !== null) window.clearTimeout(timeoutId);
    };
  }, [enabled, pollMs]);

  return systemInfo;
}
