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
    // Without it an Arc host shows no runtime row, since cuda and rocm are null.
    xpu: string | null;
    transformers: string | null;
    unsloth: string | null;
    llamaCpp: string | null;
    // `null` until the authoritative response lands, so export is never briefly enabled.
    exportSupported: boolean | null;
    exportUnsupportedReason: string | null;
    exportUnsupportedMessage: string | null;
    // False only on Windows ROCm without a loadable torchao; absent from older backends = true.
    torchaoExportSupported: boolean;
    // `null` until loaded and on older backends; only an explicit `false` hides the generator.
    videoSupported: boolean | null;
    videoUnsupportedReason: string | null;
    videoUnsupportedMessage: string | null;
    loaded: boolean;
}

const DEFAULT: HardwareInfo = {
    gpuName: null,
    vramTotalGb: null,
    vramFreeGb: null,
    gpus: [],
    torch: null,
    cuda: null,
    rocm: null,
    xpu: null,
    transformers: null,
    unsloth: null,
    llamaCpp: null,
    exportSupported: null,
    exportUnsupportedReason: null,
    exportUnsupportedMessage: null,
    torchaoExportSupported: true,
    videoSupported: null,
    videoUnsupportedReason: null,
    videoUnsupportedMessage: null,
    loaded: false,
};

const RETRY_MS = 3000;

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
                xpu: data?.versions?.xpu ?? null,
                transformers: data?.versions?.transformers ?? null,
                unsloth: data?.versions?.unsloth ?? null,
                llamaCpp: data?.llama_cpp ?? null,
                exportSupported: data?.export_supported ?? null,
                exportUnsupportedReason: data?.export_unsupported_reason ?? null,
                exportUnsupportedMessage: data?.export_unsupported_message ?? null,
                torchaoExportSupported: data?.torchao_export_supported ?? true,
                videoSupported: data?.video_supported ?? null,
                videoUnsupportedReason: data?.video_unsupported_reason ?? null,
                videoUnsupportedMessage: data?.video_unsupported_message ?? null,
                loaded: true,
            };
            if (generation === cacheGeneration) {
                cached = info;
                notifyHardwareInfo(info);
                return info;
            }
            // Superseded, so it must not become the cache, but it is still a real 200 for its callers.
            return cached ?? info;
        } catch {
            // Reset so later calls retry (e.g. backend wasn't ready).
            if (generation === cacheGeneration) fetchPromise = null;
            return DEFAULT;
        }
    })();

    return fetchPromise;
}

export function useHardwareInfo(): HardwareInfo {
    const [info, setInfo] = useState<HardwareInfo>(cached ?? DEFAULT);

    useEffect(() => {
        let cancelled = false;
        const listener = (hw: HardwareInfo) => {
            if (!cancelled) setInfo(hw);
        };

        listeners.add(listener);
        // Retry a failed probe, or pages gated on `loaded` wait out the session on one blip.
        let retry: ReturnType<typeof setTimeout> | undefined;
        const load = () => {
            fetchOnce().then((hw) => {
                listener(hw);
                if (!cancelled && !hw.loaded) retry = setTimeout(load, RETRY_MS);
            });
        };
        // The listener joins after render, so a probe that resolved in between never reached it.
        if (cached) listener(cached);
        else load();
        return () => {
            cancelled = true;
            listeners.delete(listener);
            if (retry !== undefined) clearTimeout(retry);
        };
    }, []);

    return info;
}
