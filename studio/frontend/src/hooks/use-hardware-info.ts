// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";

export interface HardwareInfo {
    gpuName: string | null;
    vramTotalGb: number | null;
    torch: string | null;
    cuda: string | null;
    transformers: string | null;
    unsloth: string | null;
}

const DEFAULT: HardwareInfo = {
    gpuName: null,
    vramTotalGb: null,
    torch: null,
    cuda: null,
    transformers: null,
    unsloth: null,
};

// Module-level cache so multiple components share one fetch.
let cached: HardwareInfo | null = null;
let fetchPromise: Promise<HardwareInfo> | null = null;

async function fetchOnce(): Promise<HardwareInfo> {
    if (cached) return cached;
    if (fetchPromise) return fetchPromise;

    fetchPromise = (async () => {
        try {
            const res = await authFetch("/api/system/hardware");
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const info: HardwareInfo = {
                gpuName: data?.gpu?.gpu_name ?? null,
                vramTotalGb: data?.gpu?.vram_total_gb ?? null,
                torch: data?.versions?.torch ?? null,
                cuda: data?.versions?.cuda ?? null,
                transformers: data?.versions?.transformers ?? null,
                unsloth: data?.versions?.unsloth ?? null,
            };
            cached = info;
            return info;
        } catch {
            // Reset promise so subsequent calls retry (e.g. backend wasn't ready)
            fetchPromise = null;
            return DEFAULT;
        }
    })();

    return fetchPromise;
}

/**
 * Fetch hardware info from `GET /api/system/hardware`.
 *
 * The result is cached at module level — only one network request is made
 * regardless of how many components call this hook.
 */
export function useHardwareInfo(): HardwareInfo {
    const [info, setInfo] = useState<HardwareInfo>(cached ?? DEFAULT);

    useEffect(() => {
        if (cached) return;

        let cancelled = false;
        fetchOnce().then((hw) => {
            if (!cancelled) setInfo(hw);
        });
        return () => { cancelled = true; };
    }, []);

    return info;
}
