// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { fetchSystemInfo } from "@/hooks/use-system";
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

export interface MemoryEstimateState {
  estimate: MemoryEstimate | null;
  /** First fetch only; re-prices set `stale` and keep old numbers so the row never blinks. */
  loading: boolean;
  stale: boolean;
}

/** Settings the backend ignores stay out, or the row re-fetches for nothing. */
function estimateKey(request: MemoryEstimateRequest | null): string | null {
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
    request.nParallel ?? null,
    request.nBatch ?? null,
    request.nUbatch ?? null,
    request.ctxCheckpoints ?? null,
    request.speculativeType ?? null,
    request.specDraftNMax ?? null,
    request.specDraftCacheType ?? null,
    request.tensorParallel ?? false,
    request.disableVision ?? false,
    request.gpuMemoryMode ?? null,
    request.gpuLayers ?? null,
    request.nCpuMoe ?? null,
    request.selectedGpuIds ?? null,
    request.llamaExtraArgs ?? null,
  ]);
}

/** In-flight requests abort when settings move, so a slow answer cannot overwrite a newer one. */
export function useMemoryEstimate(
  request: MemoryEstimateRequest | null,
  { refreshMemory = false }: { refreshMemory?: boolean } = {},
): MemoryEstimateState {
  const key = estimateKey(request);
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
    probeKey: string | null;
  }>({
    estimate: null,
    loading: false,
    stale: false,
    identity: null,
    probeKey: null,
  });
  // Ref so the effect depends on the key alone; `request` is a fresh object every render.
  const latestRequest = useRef(request);
  latestRequest.current = request;
  // A model switch, including a quant switch on the same path, must clear the numbers.
  const shownModel = useRef<string | null>(null);

  useEffect(() => {
    const pending = latestRequest.current;
    if (key == null || pending == null) {
      shownModel.current = null;
      setState({ estimate: null, loading: false, stale: false, identity: null, probeKey: null });
      return;
    }
    const identity = resolveEstimateSourceIdentity(
      pending.modelPath,
      pending.ggufVariant,
      tokenIdentity(pending.hfToken),
      pending.nativePathToken,
    );
    const modelChanged = shownModel.current !== identity;
    setState((current) =>
      modelChanged
        ? { estimate: null, loading: true, stale: false, identity, probeKey: null }
        : { ...current, loading: current.estimate == null, stale: true, identity },
    );
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetchMemoryEstimate(pending, controller.signal)
        .then(async (estimate) => {
          if (controller.signal.aborted) return;
          if (refreshMemory && estimate.available) {
            const probe = await fetchSystemInfo({ refreshMemory: true });
            if (!probe?.memory_refreshed) throw new Error("Memory probe unavailable");
          }
          if (controller.signal.aborted) return;
          shownModel.current = identity;
          setState({ estimate, loading: false, stale: false, identity, probeKey: refreshMemory ? key : null });
        })
        .catch(() => {
          if (controller.signal.aborted) return;
          shownModel.current = identity;
          setState({ estimate: null, loading: false, stale: false, identity, probeKey: null });
        });
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key, refreshMemory]);

  if (state.identity !== currentIdentity) {
    return { estimate: null, loading: currentIdentity != null, stale: false };
  }
  return {
    estimate: state.estimate,
    loading: state.loading,
    stale: state.stale || (refreshMemory && state.probeKey !== key),
  };
}
