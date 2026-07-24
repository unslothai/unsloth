// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";

export interface GpuDevice {
  index?: number;
  index_kind?: string;
  visible_ordinal?: number;
  name?: string;
  memory_total_gb?: number;
  vram_used_gb?: number;
  vram_free_gb?: number;
  vram_utilization_pct?: number | null;
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
  gpu: {
    available: boolean;
    backend?: string;
    /** Whether GGUF loads accept the device indices declared by index_kind. */
    gguf_gpu_ids_supported?: boolean;
    backend_cuda_visible_devices?: string | null;
    parent_visible_gpu_ids?: number[];
    index_kind?: string;
    devices: GpuDevice[];
  };
  ml_packages: {
    torch?: string;
    transformers?: string;
  };
}

let cachedSystem: SystemInfoResponse | null = null;
let systemFetchPromise: Promise<SystemInfoResponse> | null = null;

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

async function fetchSystemOnce({
  force = false,
}: { force?: boolean } = {}): Promise<SystemInfoResponse> {
  if (systemFetchPromise) return systemFetchPromise;
  if (!force && cachedSystem) return cachedSystem;

  systemFetchPromise = (async () => {
    try {
      const res = await authFetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      cachedSystem = data as SystemInfoResponse;
      return cachedSystem;
    } catch {
      cachedSystem = null;
      return DEFAULT_SYSTEM;
    } finally {
      systemFetchPromise = null;
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
      void fetchSystemOnce({ force })
        .then((info) => {
          if (!cancelled) setSystemInfo(info);
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
