// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
    createSingleFlight,
    startSerialPolling,
} from "@/hooks/gpu-utilization-polling";
import { useEffect, useState } from "react";

export interface GpuUtilization {
    available: boolean;
    backend: string | null;
    devices?: GpuUtilization[];
    index?: number;
    visible_ordinal?: number;
    gpu_utilization_pct: number | null;
    temperature_c: number | null;
    vram_used_gb: number | null;
    vram_total_gb: number | null;
    vram_utilization_pct: number | null;
    power_draw_w: number | null;
    power_limit_w: number | null;
    power_utilization_pct: number | null;
}

const DEFAULT: GpuUtilization = {
    available: false,
    backend: null,
    gpu_utilization_pct: null,
    temperature_c: null,
    vram_used_gb: null,
    vram_total_gb: null,
    vram_utilization_pct: null,
    power_draw_w: null,
    power_limit_w: null,
    power_utilization_pct: null,
};

const fetchGpuUtilization = createSingleFlight(async () => {
    const response = await authFetch("/api/train/hardware");
    if (!response.ok) return null;
    return (await response.json()) as GpuUtilization;
});

export function useGpuUtilization(
    enabled: boolean,
    intervalMs = 10_000,
): GpuUtilization {
    const [data, setData] = useState<GpuUtilization>(DEFAULT);

    useEffect(() => {
        if (!enabled) {
            let cancelled = false;
            queueMicrotask(() => {
                if (!cancelled) setData(DEFAULT);
            });
            return () => {
                cancelled = true;
            };
        }

        let cancelled = false;

        const stopPolling = startSerialPolling(async () => {
            try {
                const nextData = await fetchGpuUtilization();
                if (!cancelled && nextData) setData(nextData);
            } catch {
                return;
            }
        }, intervalMs);

        return () => {
            cancelled = true;
            stopPolling();
        };
    }, [enabled, intervalMs]);

    return data;
}
