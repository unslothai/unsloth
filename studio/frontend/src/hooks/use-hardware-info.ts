// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";

export interface GpuDevice {
    name: string | null;
    vramTotalGb: number | null;
}

interface ApiGpu {
    name?: string | null;
    vram_total_gb?: number | null;
}

export interface HardwareInfo {
    gpuName: string | null;
    vramTotalGb: number | null;
    vramFreeGb: number | null;
    gpus: GpuDevice[];
    torch: string | null;
    cuda: string | null;
    rocm: string | null;
    transformers: string | null;
    unsloth: string | null;
    llamaCpp: string | null;
}

const DEFAULT: HardwareInfo = {
    gpuName: null,
    vramTotalGb: null,
    vramFreeGb: null,
    gpus: [],
    torch: null,
    cuda: null,
    rocm: null,
    transformers: null,
    unsloth: null,
    llamaCpp: null,
};

// Module-level cache so multiple components share one fetch.
let cached: HardwareInfo | null = null;
let fetchPromise: Promise<HardwareInfo> | null = null;
let cacheGeneration = 0;
const listeners = new Set<(info: HardwareInfo) => void>();

function notifyHardwareInfo(info: HardwareInfo) {
    listeners.forEach((listener) => listener(info));
}

export function invalidateHardwareInfo() {
    cacheGeneration += 1;
    cached = null;
    fetchPromise = null;
}

export async function refreshHardwareInfo(): Promise<HardwareInfo> {
    invalidateHardwareInfo();
    return fetchOnce();
}

async function fetchOnce(): Promise<HardwareInfo> {
    if (cached) return cached;
    if (fetchPromise) return fetchPromise;

    const generation = cacheGeneration;
    fetchPromise = (async () => {
        try {
            const res = await authFetch("/api/system/hardware?include_details=true");
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const info: HardwareInfo = {
                gpuName: data?.gpu?.gpu_name ?? null,
                vramTotalGb: data?.gpu?.vram_total_gb ?? null,
                vramFreeGb: data?.gpu?.vram_free_gb ?? null,
                gpus: Array.isArray(data?.gpus)
                    ? data.gpus.map((g: ApiGpu) => ({
                        name: g?.name ?? null,
                        vramTotalGb: g?.vram_total_gb ?? null,
                    }))
                    : [],
                torch: data?.versions?.torch ?? null,
                cuda: data?.versions?.cuda ?? null,
                rocm: data?.versions?.rocm ?? null,
                transformers: data?.versions?.transformers ?? null,
                unsloth: data?.versions?.unsloth ?? null,
                llamaCpp: data?.llama_cpp ?? null,
            };
            if (generation === cacheGeneration) {
                cached = info;
                notifyHardwareInfo(info);
                return info;
            }
            return cached ?? DEFAULT;
        } catch {
            // Reset so subsequent calls retry (e.g. backend wasn't ready).
            if (generation === cacheGeneration) fetchPromise = null;
            return DEFAULT;
        }
    })();

    return fetchPromise;
}

/**
 * Fetch hardware info from `GET /api/system/hardware`. Cached at module level,
 * so only one request is made regardless of how many components call this hook.
 */
export function useHardwareInfo(): HardwareInfo {
    const [info, setInfo] = useState<HardwareInfo>(cached ?? DEFAULT);

    useEffect(() => {
        let cancelled = false;
        const listener = (hw: HardwareInfo) => {
            if (!cancelled) setInfo(hw);
        };

        listeners.add(listener);
        if (!cached) fetchOnce().then(listener);
        return () => {
            cancelled = true;
            listeners.delete(listener);
        };
    }, []);

    return info;
}
