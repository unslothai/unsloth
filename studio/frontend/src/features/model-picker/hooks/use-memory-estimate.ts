// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import {
  type MemoryEstimate,
  type MemoryEstimateRequest,
  fetchMemoryEstimate,
} from "../api/memory-estimate";
import {
  resolveEstimateSourceIdentity,
  resolveTokenIdentity as tokenIdentity,
} from "../model-config/estimate-context";

/** Long enough that a slider drag lands one request, not sixty; short enough that letting go
 *  feels immediate. The fetch itself is a header walk, tens of ms. */
const ESTIMATE_DEBOUNCE_MS = 250;

export interface MemoryEstimateState {
  estimate: MemoryEstimate | null;
  /** First fetch for this model, nothing to show yet. A re-price keeps the old numbers up and sets
   *  `stale` instead, so the row never blinks on a slider step. */
  loading: boolean;
  /** The numbers shown are for older settings; a fresh answer is on its way. */
  stale: boolean;
}

/** Everything that changes the answer. Settings the backend ignores stay out, or the row re-fetches for nothing. */
function estimateKey(request: MemoryEstimateRequest | null): string | null {
  if (!request) return null;
  return JSON.stringify([
    request.modelPath,
    request.ggufVariant ?? null,
    tokenIdentity(request.hfToken),
    request.nativePathToken ?? null,
    request.nCtx ?? null,
    request.cacheTypeKv ?? null,
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

/** Debounced memory estimate for a prospective load. Pass null to stand down. In-flight requests
 *  abort when the settings move again, so a slow answer for a context already dragged past
 *  cannot overwrite a newer one. */
export function useMemoryEstimate(
  request: MemoryEstimateRequest | null,
): MemoryEstimateState {
  const key = estimateKey(request);
  // Computed during render, not read from a ref the effect updates after paint: the effect below
  // still clears on a switch, but it runs after React has painted, so a direct switch between
  // two GGUFs showed the previous model's footprint under the new name for a frame.
  const currentIdentity =
    request == null
      ? null
      : resolveEstimateSourceIdentity(
          request.modelPath,
          request.ggufVariant,
          tokenIdentity(request.hfToken),
          request.nativePathToken,
        );
  const [state, setState] = useState<MemoryEstimateState & { identity: string | null }>({
    estimate: null,
    loading: false,
    stale: false,
    identity: null,
  });
  // Read inside the effect so it depends on the key alone: `request` is a fresh object every
  // render and would restart the debounce on any keystroke.
  const latestRequest = useRef(request);
  latestRequest.current = request;
  // Which model the numbers belong to. A switch must clear them: one model's footprint under
  // another's name is worse than none. The quantization counts as a switch -- Q4_K_M to F16 on
  // one repository leaves modelPath alone while the weights quadruple.
  const shownModel = useRef<string | null>(null);

  useEffect(() => {
    const pending = latestRequest.current;
    if (key == null || pending == null) {
      shownModel.current = null;
      setState({ estimate: null, loading: false, stale: false, identity: null });
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
        ? { estimate: null, loading: true, stale: false, identity }
        : { ...current, loading: current.estimate == null, stale: true, identity },
    );
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetchMemoryEstimate(pending, controller.signal)
        .then((estimate) => {
          if (controller.signal.aborted) return;
          shownModel.current = identity;
          setState({ estimate, loading: false, stale: false, identity });
        })
        .catch(() => {
          if (controller.signal.aborted) return;
          // A failed estimate is not a failed panel; drop the row.
          shownModel.current = identity;
          setState({ estimate: null, loading: false, stale: false, identity });
        });
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key]);

  // State that belongs to a different source is not shown at all, not even for the frame before the effect clears it.
  if (state.identity !== currentIdentity) {
    return { estimate: null, loading: currentIdentity != null, stale: false };
  }
  return { estimate: state.estimate, loading: state.loading, stale: state.stale };
}
