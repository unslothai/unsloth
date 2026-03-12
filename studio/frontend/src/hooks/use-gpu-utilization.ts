// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useRef, useState } from "react";

export interface GpuUtilization {
    available: boolean;
    backend: string | null;
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

/**
 * Poll `GET /api/train/hardware` for live GPU utilization stats.
 *
 * Only polls while `enabled` is true (i.e. training is running).
 * Polling interval defaults to 10 000 ms.
 */
export function useGpuUtilization(
    enabled: boolean,
    intervalMs = 10_000,
): GpuUtilization {
    const [data, setData] = useState<GpuUtilization>(DEFAULT);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    useEffect(() => {
        if (!enabled) {
            // Reset when training stops so the cards show "--" again
            setData(DEFAULT);
            return;
        }

        let cancelled = false;

        async function poll() {
            try {
                const res = await authFetch("/api/train/hardware");
                if (!res.ok || cancelled) return;
                const json = (await res.json()) as GpuUtilization;
                if (!cancelled) setData(json);
            } catch {
                // Silently ignore — next poll will retry
            }
        }

        // Fetch immediately, then set up interval
        void poll();
        timerRef.current = setInterval(() => void poll(), intervalMs);

        return () => {
            cancelled = true;
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, [enabled, intervalMs]);

    return data;
}
