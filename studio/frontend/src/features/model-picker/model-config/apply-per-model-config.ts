// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  GPU_LAYERS_AUTO,
  defaultInferenceParams,
  normalizeSpeculativeType,
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  reconcilePersistedGpuIds,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  cachedPinnableGpuIndices,
  ensureGpuDeviceCache,
} from "@/hooks/use-gpu-info";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
  normalizeMaxSeqLength,
} from "./per-model-config";

function cleanTemplate(value: string | null | undefined): string | null {
  return value?.trim() ? value : null;
}

export function applyPerModelConfigToRuntime(config: PerModelConfig): void {
  // Fall back to the standing default when the model has no saved
  // maxSeqLength. maxSeqLength is the only per-model field carried on
  // params (the rest are reset below), so without this a model with no
  // remembered config would inherit the previously loaded model's value.
  const maxSeqLength =
    normalizeMaxSeqLength(config.maxSeqLength) ??
    defaultInferenceParams.maxSeqLength;
  const store = useChatRuntimeStore.getState();
  if (maxSeqLength !== store.params.maxSeqLength) {
    store.setParams({ ...store.params, maxSeqLength });
  }
  useChatRuntimeStore.setState({
    customContextLength: config.customContextLength ?? null,
    kvCacheDtype: config.kvCacheDtype ?? null,
    speculativeType:
      normalizeSpeculativeType(config.speculativeType) ??
      readPersistedSpeculativeType(),
    specDraftNMax: config.specDraftNMax ?? null,
    tensorParallel: config.tensorParallel ?? false,
    chatTemplateOverride: cleanTemplate(config.chatTemplateOverride),
    // GPU Memory knobs are per-model (GGUF-only). Absent = defaults; the mode is
    // a standing preference so an absent mode falls back to the persisted one.
    // The per-GPU split ratio is never remembered, so it always resets. The GPU
    // pick is reconciled against the GPUs present now (a saved [1] on a 1-GPU
    // host would otherwise be sent and rejected).
    gpuMemoryMode: config.gpuMemoryMode ?? readPersistedGpuMemoryMode(),
    gpuLayers: config.gpuLayers ?? GPU_LAYERS_AUTO,
    nCpuMoe: config.nCpuMoe ?? 0,
    splitRatio: null,
    selectedGpuIds:
      config.selectedGpuIds !== undefined
        ? reconcilePersistedGpuIds(config.selectedGpuIds, {
            fromPersisted: true,
          })
        : null,
  });

  // reconcilePersistedGpuIds can only clear a wrong-space pick (physical
  // CUDA/ROCm ids vs ggml Vulkan ordinals, after a llama.cpp backend swap) once
  // the GPU cache is warm. This runs synchronously on model selection, which can
  // happen before any GPU hook has fetched /api/system -- on a cold cache the
  // reconcile passes the saved ids through unvalidated, and if it then launders
  // into the store the later load path reconciles it as a live pick
  // (fromPersisted false) and never clears it, pinning the wrong card. So when
  // the pick was applied cold, warm the cache and re-reconcile, overwriting only
  // if the user has not replaced the pick in the meantime.
  const persistedIds = config.selectedGpuIds;
  if (persistedIds != null && cachedPinnableGpuIndices() === null) {
    const applied = useChatRuntimeStore.getState().selectedGpuIds;
    void ensureGpuDeviceCache().then(() => {
      const store = useChatRuntimeStore.getState();
      if (store.selectedGpuIds !== applied) return; // user changed it since
      const revalidated = reconcilePersistedGpuIds(persistedIds, {
        fromPersisted: true,
      });
      if (revalidated !== store.selectedGpuIds) {
        useChatRuntimeStore.setState({ selectedGpuIds: revalidated });
      }
    });
  }
}

export function applyModelLoadConfigToRuntime(
  config: PerModelConfig | null | undefined,
): boolean {
  const hasConfig = config != null;
  applyPerModelConfigToRuntime(config ?? DEFAULT_PER_MODEL_CONFIG);
  return hasConfig;
}

export function currentRuntimePerModelConfig(
  options: { includeMaxSeqLength?: boolean } = {},
): PerModelConfig {
  const s = useChatRuntimeStore.getState();
  return {
    customContextLength: s.customContextLength ?? null,
    maxSeqLength: options.includeMaxSeqLength
      ? normalizeMaxSeqLength(s.params.maxSeqLength)
      : null,
    kvCacheDtype: s.kvCacheDtype ?? null,
    speculativeType: normalizeSpeculativeType(s.speculativeType),
    specDraftNMax: s.specDraftNMax ?? null,
    tensorParallel: s.tensorParallel ?? false,
    chatTemplateOverride: cleanTemplate(s.chatTemplateOverride),
    // Snapshot the live GPU knobs too so a failed switch rolls the previous
    // model's GPU Memory settings back (see applyPerModelConfigToRuntime). The
    // split ratio is intentionally never remembered.
    gpuMemoryMode: s.gpuMemoryMode,
    gpuLayers: s.gpuLayers,
    nCpuMoe: s.nCpuMoe,
    selectedGpuIds: s.selectedGpuIds,
  };
}

export function perModelConfigsEqual(
  a: PerModelConfig,
  b: PerModelConfig,
): boolean {
  return (
    (a.customContextLength ?? null) === (b.customContextLength ?? null) &&
    normalizeMaxSeqLength(a.maxSeqLength) ===
      normalizeMaxSeqLength(b.maxSeqLength) &&
    (a.kvCacheDtype ?? null) === (b.kvCacheDtype ?? null) &&
    normalizeSpeculativeType(a.speculativeType) ===
      normalizeSpeculativeType(b.speculativeType) &&
    (a.specDraftNMax ?? null) === (b.specDraftNMax ?? null) &&
    Boolean(a.tensorParallel) === Boolean(b.tensorParallel) &&
    cleanTemplate(a.chatTemplateOverride) ===
      cleanTemplate(b.chatTemplateOverride) &&
    gpuFieldsEqual(a, b)
  );
}

// Serialize the per-model GPU knobs with the same "absent == default"
// coalescing the store applies: mode auto/absent, gpuLayers Auto (< 0) /
// absent, nCpuMoe 0 / absent, and the GPU pick (null / absent = all GPUs).
export function gpuFieldsSignature(config: PerModelConfig): string {
  return [
    config.gpuMemoryMode ?? "auto",
    config.gpuLayers == null || config.gpuLayers < 0 ? -1 : config.gpuLayers,
    config.nCpuMoe ?? 0,
    config.selectedGpuIds == null
      ? "all"
      : [...config.selectedGpuIds].sort((a, b) => a - b).join(","),
  ].join("|");
}

function gpuFieldsEqual(a: PerModelConfig, b: PerModelConfig): boolean {
  return gpuFieldsSignature(a) === gpuFieldsSignature(b);
}
