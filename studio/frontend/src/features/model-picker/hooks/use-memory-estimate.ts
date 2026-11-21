// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import type { SystemGpuDevice } from "@/hooks/gpu-selection";
import { fetchSystemInfo } from "@/hooks/use-system";
import { extraArgFlags, tensorSplitOverrideCount } from "../model-config/llama-extra-args";
import {
  type MemoryEstimate,
  type MemoryEstimateRequest,
  fetchMemoryEstimate,
} from "../api/memory-estimate";
import {
  resolveEstimateSourceIdentity,
  resolveTokenIdentity as tokenIdentity,
} from "../model-config/estimate-context";

const ESTIMATE_DEBOUNCE_MS = 250;
// The backend accepts samples up to 10 s old. Leave time for the debounced estimate to read them.
const SPLIT_PROBE_INTERVAL_MS = 5000;
const SPLIT_PROBE_MAX_AGE_MS = 3000;

let splitProbe: { started: number; pending: ReturnType<typeof fetchSystemInfo> } | null = null;
function refreshSplitProbe() {
  const now = performance.now();
  if (!splitProbe || now - splitProbe.started >= SPLIT_PROBE_INTERVAL_MS) {
    splitProbe = {
      started: now,
      pending: fetchSystemInfo({ force: true, gpuMaxAgeMs: SPLIT_PROBE_MAX_AGE_MS }),
    };
  }
  return splitProbe.pending;
}

const ESTIMATE_CACHE_LIMIT = 32;

// Survives remounts (the Model dropdown remounts the page), so reopening paints the last numbers at once.
const estimateCache = new Map<string, MemoryEstimate>();

export function cachedMemoryEstimate(key: string | null): MemoryEstimate | null {
  return key == null ? null : (estimateCache.get(key) ?? null);
}

export function rememberMemoryEstimate(key: string, estimate: MemoryEstimate): void {
  estimateCache.delete(key);
  estimateCache.set(key, estimate);
  if (estimateCache.size > ESTIMATE_CACHE_LIMIT) {
    estimateCache.delete(estimateCache.keys().next().value as string);
  }
}

export interface MemoryEstimateState {
  estimate: MemoryEstimate | null;
  /** First fetch only; re-prices set `stale` and keep old numbers so the row never blinks. */
  loading: boolean;
  stale: boolean;
}

function splitInventory(
  request: MemoryEstimateRequest | null,
  devices: SystemGpuDevice[] | undefined,
) {
  if (!devices || !request || request.gpuMemoryMode !== "manual" || request.nCpuMoe) return null;
  const ids = request.selectedGpuIds ?? devices.map((device) => device.index);
  if (ids.length < 2) return null;
  const args = request.llamaExtraArgs ?? [];
  const flags = extraArgFlags(args);
  if (request.tensorParallel && !flags.includes("--split-mode") && !flags.includes("-sm")) return null;
  const layerOverride = flags.some((flag) =>
    ["-ngl", "--gpu-layers", "--n-gpu-layers"].includes(flag),
  );
  if ((request.gpuLayers ?? 0) <= 0 && !layerOverride) return null;
  // A mismatched --tensor-split is dropped at launch, leaving the free-memory split.
  const shareCount = tensorSplitOverrideCount(args) ?? request.tensorSplit?.length;
  const usesFreeMemory = shareCount !== ids.length;
  return {
    usesFreeMemory,
    devices: ids.map((id) => {
      const gpu = devices.find((device) => device.index === id);
      if (!gpu) return null;
      return [
        id, gpu.indexKind, gpu.sharedMemory || gpu.unifiedMemory === true,
        usesFreeMemory && (gpu.memoryFreeKnown ?? gpu.memoryFreeGb > 0) ? gpu.memoryFreeGb : null,
      ];
    }),
  };
}

/** Settings the backend ignores stay out, or the row re-fetches for nothing. */
export function estimateKey(
  request: MemoryEstimateRequest | null,
  inventory: ReturnType<typeof splitInventory> = null,
): string | null {
  if (!request) return null;
  return JSON.stringify([
    request.modelPath,
    request.ggufVariant ?? null,
    tokenIdentity(request.hfToken),
    request.nativePathToken ?? null,
    request.nCtx ?? null,
    request.cacheTypeKv ?? null,
    request.maxSeqLength ?? null,
    request.mlxKvQuant ?? null,
    request.mlxPleOffload ?? null,
    request.mlxExpertOffload ?? null,
    request.nParallel ?? null,
    request.nBatch ?? null,
    request.nUbatch ?? null,
    request.ctxCheckpoints ?? null,
    request.cacheRam ?? null,
    request.speculativeType ?? null,
    request.specDraftNMax ?? null,
    request.specDraftModel ?? null,
    request.specDraftCacheType ?? null,
    request.tensorParallel ?? false,
    request.disableVision ?? false,
    request.gpuMemoryMode ?? null,
    request.gpuLayers ?? null,
    request.nCpuMoe ?? null,
    request.selectedGpuIds ?? null,
    request.tensorSplit ?? null,
    request.llamaExtraArgs ?? null,
    inventory,
  ]);
}

/** In-flight requests abort when settings move, so a slow answer cannot overwrite a newer one. */
export function useMemoryEstimate(
  request: MemoryEstimateRequest | null,
  {
    refreshMemory = false,
    gpuDevices,
  }: { refreshMemory?: boolean; gpuDevices?: SystemGpuDevice[] } = {},
): MemoryEstimateState {
  const inventory = splitInventory(request, gpuDevices);
  const refreshSplitMemory = inventory?.usesFreeMemory === true;
  const settings = estimateKey(request, inventory);
  const key = refreshMemory && settings !== null ? `probe:${settings}` : settings;
  // Computed during render: the clearing effect runs after paint and would flash the old model's numbers.
  const currentIdentity =
    request == null
      ? null
      : resolveEstimateSourceIdentity(
          request.modelPath,
          request.ggufVariant,
          tokenIdentity(request.hfToken),
          request.nativePathToken,
        );
  const [state, setState] = useState<MemoryEstimateState & {
    identity: string | null;
    estimateKey: string | null;
  }>(() => {
    const cached = refreshMemory ? null : cachedMemoryEstimate(key);
    return {
      estimate: cached,
      loading: false,
      stale: cached != null,
      identity: cached ? currentIdentity : null,
      estimateKey: null,
    };
  });
  // Ref so the effect depends on the key alone; `request` is a fresh object every render.
  const latestRequest = useRef(request);
  latestRequest.current = request;
  // A model switch, including a quant switch on the same path, must clear the numbers.
  const shownModel = useRef<string | null>(state.estimate ? currentIdentity : null);
  const firstDispatch = useRef<string | null>(null);

  useEffect(() => {
    const pending = latestRequest.current;
    if (key == null || pending == null) {
      shownModel.current = null;
      setState({ estimate: null, loading: false, stale: false, identity: null, estimateKey: null });
      return;
    }
    const identity = resolveEstimateSourceIdentity(
      pending.modelPath,
      pending.ggufVariant,
      tokenIdentity(pending.hfToken),
      pending.nativePathToken,
    );
    const modelChanged = shownModel.current !== identity;
    const cached = refreshMemory ? null : cachedMemoryEstimate(key);
    if (cached) shownModel.current = identity;
    setState((current) =>
      cached
        ? { ...current, estimate: cached, loading: false, stale: true, identity }
        : modelChanged
          ? { estimate: null, loading: true, stale: false, identity, estimateKey: null }
          : { ...current, loading: current.estimate == null, stale: true, identity },
    );
    const controller = new AbortController();
    // Only a model's first request skips the debounce; edits made while it runs still settle first.
    const immediate = modelChanged && !cached && firstDispatch.current !== identity;
    if (immediate) firstDispatch.current = identity;
    const timer = setTimeout(async () => {
      if (refreshSplitMemory) {
        await refreshSplitProbe();
        if (controller.signal.aborted) return;
      }
      fetchMemoryEstimate(pending, controller.signal)
        .then(async (estimate) => {
          if (controller.signal.aborted) return;
          if (refreshMemory && estimate.available) {
            const probe = await fetchSystemInfo({ refreshMemory: true });
            if (!probe?.memory_refreshed) throw new Error("Memory probe unavailable");
          }
          if (controller.signal.aborted) return;
          shownModel.current = identity;
          if (estimate.available) rememberMemoryEstimate(key, estimate);
          setState({ estimate, loading: false, stale: false, identity, estimateKey: key });
        })
        .catch(() => {
          if (controller.signal.aborted) return;
          shownModel.current = identity;
          setState({ estimate: null, loading: false, stale: false, identity, estimateKey: null });
        });
    }, immediate ? 0 : ESTIMATE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key, refreshMemory, refreshSplitMemory]);

  if (state.identity !== currentIdentity) {
    const cached = refreshMemory ? null : cachedMemoryEstimate(key);
    return cached
      ? { estimate: cached, loading: false, stale: true }
      : { estimate: null, loading: currentIdentity != null, stale: false };
  }
  return {
    estimate: state.estimate,
    loading: state.loading,
    stale: state.stale || ((state.estimate !== null || refreshMemory) && state.estimateKey !== key),
  };
}
