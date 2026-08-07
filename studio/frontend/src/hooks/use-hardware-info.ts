


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
    // Intel XPU (SYCL). The backend always emitted this; without it the About tab shows no
    // runtime row at all on an Arc host, where cuda and rocm are both null.
    xpu: string | null;
    transformers: string | null;
    unsloth: string | null;
    llamaCpp: string | null;
    // Whether export can run here (true only on a supported accelerator), with a torch-aware
    // reason. `null` until the authoritative response lands, so callers don't briefly enable
    // export; `loaded` flips true once a real (non-error) response arrives.
    exportSupported: boolean | null;
    exportUnsupportedReason: string | null;
    exportUnsupportedMessage: string | null;
    // Whether video generation can run here. Same tri-state as export: `null` until the
    // authoritative response lands, and `null` too against a backend that predates the field,
    // so only an explicit `false` hides the generator.
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
    videoSupported: null,
    videoUnsupportedReason: null,
    videoUnsupportedMessage: null,
    loaded: false,
};

// How long a caller waits before re-probing after a failed read. See useHardwareInfo.
const RETRY_MS = 3000;

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
                xpu: data?.versions?.xpu ?? null,
                transformers: data?.versions?.transformers ?? null,
                unsloth: data?.versions?.unsloth ?? null,
                llamaCpp: data?.llama_cpp ?? null,
                exportSupported: data?.export_supported ?? null,
                exportUnsupportedReason: data?.export_unsupported_reason ?? null,
                exportUnsupportedMessage: data?.export_unsupported_message ?? null,
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
            // Superseded by a later invalidate, so it must not become the cache. It is
            // still a real 200 though: returning DEFAULT tells every caller riding this
            // promise that a healthy read failed, and load() reads that as a failed probe.
            return cached ?? info;
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
        // A failed probe resolves to DEFAULT (loaded false) and clears the in-flight promise,
        // but nothing re-ran it, so the only retry was another component happening to mount.
        // Callers that gate a whole page on `loaded` would wait out the session on one blip.
        let retry: ReturnType<typeof setTimeout> | undefined;
        const load = () => {
            fetchOnce().then((hw) => {
                listener(hw);
                if (!cancelled && !hw.loaded) retry = setTimeout(load, RETRY_MS);
            });
        };
        // `info` was seeded from `cached` at render time, but this listener only joins the
        // set now. A probe that resolved in between notified the listeners registered at
        // the time -- not this one -- and left `cached` set, so `if (!cached) load()` alone
        // skipped the fetch and nothing was ever going to call setInfo. The component then
        // sat on DEFAULT with loaded false for its whole life, which on /video is the
        // capability gate's "Checking this machine for video support..." for the session.
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
