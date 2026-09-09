// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  GPU_LAYERS_AUTO,
  normalizeSpeculativeType,
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  reconcilePersistedGpuSelection,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import { defaultInferenceParams } from "@/features/chat/presets/preset-policy";
// Its own module so hosts needing only the signature skip the chat runtime store.
import { gpuFieldsSignature } from "./config-signature";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
  normalizeMaxSeqLength,
} from "./per-model-config";

export { gpuFieldsSignature };

function cleanTemplate(value: string | null | undefined): string | null {
  return value?.trim() ? value : null;
}

export function applyPerModelConfigToRuntime(
  config: PerModelConfig,
  options: { isDiffusion?: boolean } = {},
): void {
  // Fall back to the standing default when the model has no saved maxSeqLength. It is the only
  // per-model field carried on params, so without this a model with no remembered config would
  // inherit the previously loaded model's value.
  const maxSeqLength =
    normalizeMaxSeqLength(config.maxSeqLength) ??
    defaultInferenceParams.maxSeqLength;
  const store = useChatRuntimeStore.getState();
  if (maxSeqLength !== store.params.maxSeqLength) {
    store.setParams({ ...store.params, maxSeqLength });
  }
  const gpuSelection =
    config.selectedGpuIds !== undefined
      ? reconcilePersistedGpuSelection(
          config.selectedGpuIds,
          config.selectedGpuIndexKind,
          options.isDiffusion,
        )
      : { ids: null, indexKind: null };
  useChatRuntimeStore.setState({
    customContextLength: config.customContextLength ?? null,
    mlxKvBits: config.mlxKvBits ?? null,
    kvCacheDtype: config.kvCacheDtype ?? null,
    speculativeType:
      normalizeSpeculativeType(config.speculativeType) ??
      readPersistedSpeculativeType(),
    specDraftNMax: config.specDraftNMax ?? null,
    specDraftCacheDtype: config.specDraftCacheDtype ?? null,
    nParallel: config.nParallel ?? null,
    // the diffusion runner ignores the llama-server batch flags
    nBatch: options.isDiffusion ? null : (config.nBatch ?? null),
    nUbatch: options.isDiffusion ? null : (config.nUbatch ?? null),
    // Same reason as the batch flags: these are llama-server's own, and the diffusion runner never launches one.
    loadMode: options.isDiffusion ? null : (config.loadMode ?? null),
    ctxCheckpoints: options.isDiffusion ? null : (config.ctxCheckpoints ?? null),
    cacheRam: options.isDiffusion ? null : (config.cacheRam ?? null),
    tensorParallel: options.isDiffusion
      ? false
      : (config.tensorParallel ?? false),
    // The diffusion runner has no projector to skip, so the toggle is inert there for the same
    // reason tensorParallel is.
    disableVision: options.isDiffusion
      ? false
      : (config.disableVision ?? false),
    chatTemplateOverride: cleanTemplate(config.chatTemplateOverride),
    // GPU Memory knobs are per-model (GGUF-only). Absent = defaults; the mode is a standing
    // preference so an absent mode falls back to the persisted one. The per-GPU split ratio is
    // never remembered. The GPU pick is reconciled against the GPUs present now. A diffusion
    // config is sanitized to gpuMemoryMode "auto" because the mode does not apply, not because
    // the user chose Auto: writing that into the live standing preference would strand the session
    // on Auto, since the load skips saveGpuMemoryMode for diffusion and the next ordinary GGUF
    // would persist it over the user's Manual.
    gpuMemoryMode: options.isDiffusion
      ? readPersistedGpuMemoryMode()
      : (config.gpuMemoryMode ?? readPersistedGpuMemoryMode()),
    gpuLayers: config.gpuLayers ?? GPU_LAYERS_AUTO,
    nCpuMoe: config.nCpuMoe ?? 0,
    splitRatio: null,
    selectedGpuIds: gpuSelection.ids,
    selectedGpuIndexKind: gpuSelection.indexKind,
  });
}

export function applyModelLoadConfigToRuntime(
  config: PerModelConfig | null | undefined,
  options: { isDiffusion?: boolean } = {},
): boolean {
  const hasConfig = config != null;
  applyPerModelConfigToRuntime(config ?? DEFAULT_PER_MODEL_CONFIG, options);
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
    mlxKvBits: s.mlxKvBits ?? null,
    speculativeType: normalizeSpeculativeType(s.speculativeType),
    specDraftNMax: s.specDraftNMax ?? null,
    specDraftCacheDtype: s.specDraftCacheDtype ?? null,
    nParallel: s.nParallel ?? null,
    nBatch: s.nBatch ?? null,
    nUbatch: s.nUbatch ?? null,
    loadMode: s.loadMode ?? null,
    ctxCheckpoints: s.ctxCheckpoints ?? null,
    cacheRam: s.cacheRam ?? null,
    tensorParallel: s.tensorParallel ?? false,
    disableVision: s.disableVision ?? false,
    chatTemplateOverride: cleanTemplate(s.chatTemplateOverride),
    // Snapshot the live GPU knobs too so a failed switch rolls the previous model's GPU Memory
    // settings back. The split ratio is intentionally never remembered.
    gpuMemoryMode: s.gpuMemoryMode,
    gpuLayers: s.gpuLayers,
    nCpuMoe: s.nCpuMoe,
    selectedGpuIds: s.selectedGpuIds,
    selectedGpuIndexKind: s.selectedGpuIndexKind,
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
    (a.mlxKvBits ?? null) === (b.mlxKvBits ?? null) &&
    normalizeSpeculativeType(a.speculativeType) ===
      normalizeSpeculativeType(b.speculativeType) &&
    (a.specDraftNMax ?? null) === (b.specDraftNMax ?? null) &&
    (a.specDraftCacheDtype ?? null) === (b.specDraftCacheDtype ?? null) &&
    (a.nParallel ?? null) === (b.nParallel ?? null) &&
    (a.nBatch ?? null) === (b.nBatch ?? null) &&
    (a.nUbatch ?? null) === (b.nUbatch ?? null) &&
    (a.loadMode ?? null) === (b.loadMode ?? null) &&
    (a.ctxCheckpoints ?? null) === (b.ctxCheckpoints ?? null) &&
    (a.cacheRam ?? null) === (b.cacheRam ?? null) &&
    Boolean(a.tensorParallel) === Boolean(b.tensorParallel) &&
    Boolean(a.disableVision) === Boolean(b.disableVision) &&
    cleanTemplate(a.chatTemplateOverride) ===
      cleanTemplate(b.chatTemplateOverride) &&
    extraArgsSignature(a.llamaExtraArgs) === extraArgsSignature(b.llamaExtraArgs) &&
    gpuFieldsEqual(a, b)
  );
}

/** Compare on the launched command, so "not loaded" and "cleared" are equal here. They differ
 *  only in what a SAVE does, and treating them as different would make the row read as an
 *  unsaved change the moment it finished reading the server. */
function extraArgsSignature(value: string[] | null | undefined): string {
  return (value ?? []).join("\u0000");
}

function gpuFieldsEqual(a: PerModelConfig, b: PerModelConfig): boolean {
  return gpuFieldsSignature(a) === gpuFieldsSignature(b);
}
