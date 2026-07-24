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
import type { GpuIndexKind } from "@/hooks/use-gpu-info";
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
  // The pick's index space is intrinsic to the config: its
  // selectedGpuIdsIndexKind stamp (a fresh edit stamps the current kind; a
  // stored pick carries the kind it was saved under; absent = a pre-stamp
  // physical pick). reconcilePersistedGpuIds drops it only when that stamp no
  // longer matches the current backend, so a same-backend restore keeps the
  // pick and a cross-space one is cleared -- no call-site "is this persisted?"
  // guessing. On a cold cache the reconcile keeps the ids; the stamp is carried
  // into the store so the load-boundary reconcile of the live value can decide.
  const savedKind: GpuIndexKind | null =
    config.selectedGpuIds == null
      ? null
      : (config.selectedGpuIdsIndexKind ?? "physical");
  const reconciledGpuIds =
    config.selectedGpuIds === undefined
      ? null
      : reconcilePersistedGpuIds(config.selectedGpuIds, savedKind);
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
    selectedGpuIds: reconciledGpuIds,
    selectedGpuIdsKind: reconciledGpuIds == null ? null : savedKind,
  });
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
    // Carry the index space the live pick is in so a save/restore round-trip
    // (and a cancel-restore of this snapshot) can drop it after a backend swap.
    selectedGpuIdsIndexKind:
      s.selectedGpuIds == null ? undefined : (s.selectedGpuIdsKind ?? undefined),
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
