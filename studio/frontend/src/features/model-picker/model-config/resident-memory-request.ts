// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useChatRuntimeStore } from "../../chat/index.ts";
import type { MemoryEstimateRequest } from "../api/memory-estimate.ts";

type ResidentState = Pick<
  ReturnType<typeof useChatRuntimeStore.getState>,
  | "modelLoading"
  | "loadedKvCacheDtype"
  | "loadedNParallel"
  | "loadedNBatch"
  | "loadedNUbatch"
  | "loadedCtxCheckpoints"
  | "loadedSpeculativeType"
  | "loadedSpecDraftNMax"
  | "loadedSpecDraftCacheDtype"
  | "loadedTensorParallel"
  | "loadedDisableVision"
  | "loadedGpuMemoryMode"
  | "loadedGpuLayers"
  | "loadedNCpuMoe"
  | "loadedGpuIds"
  | "loadedLlamaExtraArgs"
  | "loadedCpuFallback"
  | "specFallbackReason"
  | "mmprojFallbackReason"
>;

/** Editable controls can already describe the next load. */
export function selectResidentEstimateSettings(state: ResidentState) {
  if (
    state.modelLoading ||
    state.loadedGpuMemoryMode == null ||
    state.loadedSpeculativeType == null ||
    state.loadedTensorParallel == null ||
    state.loadedDisableVision == null ||
    (state.loadedGpuMemoryMode === "manual" &&
      (state.loadedGpuLayers == null || state.loadedNCpuMoe == null)) ||
    state.loadedCpuFallback ||
    state.specFallbackReason ||
    state.mmprojFallbackReason
  ) {
    return null;
  }
  return {
    cacheTypeKv: state.loadedKvCacheDtype,
    nParallel: state.loadedNParallel,
    nBatch: state.loadedNBatch,
    nUbatch: state.loadedNUbatch,
    ctxCheckpoints: state.loadedCtxCheckpoints,
    speculativeType: state.loadedSpeculativeType,
    specDraftNMax: state.loadedSpecDraftNMax,
    specDraftCacheType: state.loadedSpecDraftCacheDtype,
    tensorParallel: state.loadedTensorParallel,
    disableVision: state.loadedDisableVision,
    gpuMemoryMode: state.loadedGpuMemoryMode,
    gpuLayers:
      state.loadedGpuLayers != null && state.loadedGpuLayers >= 0
        ? state.loadedGpuLayers
        : null,
    nCpuMoe: state.loadedNCpuMoe,
    selectedGpuIds: state.loadedGpuIds,
    llamaExtraArgs: state.loadedLlamaExtraArgs,
  };
}

/** Copy only source identity from the pending request. */
export function resolveResidentEstimateRequest(
  source: MemoryEstimateRequest | null,
  settings: ReturnType<typeof selectResidentEstimateSettings>,
  context: number | null,
): MemoryEstimateRequest | null {
  if (
    !source ||
    !settings ||
    context == null ||
    !Number.isFinite(context) ||
    context <= 0
  )
    return null;
  return {
    modelPath: source.modelPath,
    ggufVariant: source.ggufVariant,
    hfToken: source.hfToken,
    nativePathToken: source.nativePathToken,
    ...settings,
    nCtx: Math.floor(context),
  };
}
