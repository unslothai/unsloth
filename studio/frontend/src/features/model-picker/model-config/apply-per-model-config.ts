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
import { reconcileTensorSplit } from "@/hooks/gpu-tensor-split";
import { defaultInferenceParams } from "@/features/chat/presets/preset-policy";
// Separate module so hosts needing only the signature skip the chat runtime store.
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
  // Without the standing default, a model with no saved config inherits the previous model's value.
  const maxSeqLength =
    normalizeMaxSeqLength(config.maxSeqLength) ??
    defaultInferenceParams.maxSeqLength;
  const store = useChatRuntimeStore.getState();
  const engine = config.engine ?? "auto";
  const engineParallelism = config.engineParallelism ?? "tensor";
  const enginePrecision = config.enginePrecision ?? "auto";
  if (
    maxSeqLength !== store.params.maxSeqLength ||
    engine !== (store.params.engine ?? "auto") ||
    enginePrecision !== (store.params.enginePrecision ?? "auto") ||
    engineParallelism !== (store.params.engineParallelism ?? "tensor")
  ) {
    store.setParams({ ...store.params, maxSeqLength, engine, enginePrecision, engineParallelism });
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
    mlxKvQuant: config.mlxKvQuant ?? null,
    mlxInt8Prefill: config.mlxInt8Prefill ?? false,
    kvCacheDtype: config.kvCacheDtype ?? null,
    speculativeType:
      normalizeSpeculativeType(config.speculativeType) ??
      readPersistedSpeculativeType(),
    specDraftNMax: config.specDraftNMax ?? null,
    specDraftCacheDtype: config.specDraftCacheDtype ?? null,
    specDraftModel: config.specDraftModel ?? null,
    nParallel: config.nParallel ?? null,
    reasoningBudget: options.isDiffusion ? -1 : config.reasoningBudget,
    reasoningBudgetMessage: options.isDiffusion
      ? ""
      : config.reasoningBudgetMessage,
    // The diffusion runner ignores the llama-server batch flags.
    nBatch: options.isDiffusion ? null : (config.nBatch ?? null),
    nUbatch: options.isDiffusion ? null : (config.nUbatch ?? null),
    loadMode: options.isDiffusion ? null : (config.loadMode ?? null),
    ctxCheckpoints: options.isDiffusion ? null : (config.ctxCheckpoints ?? null),
    cacheRam: options.isDiffusion ? null : (config.cacheRam ?? null),
    tensorParallel: options.isDiffusion
      ? false
      : (config.tensorParallel ?? false),
    disableVision: options.isDiffusion
      ? false
      : (config.disableVision ?? false),
    chatTemplateOverride: cleanTemplate(config.chatTemplateOverride),
    // GPU Memory knobs are per-model (GGUF-only). Absent = defaults; the mode is a standing
    // preference so an absent mode falls back to the persisted one. The per-GPU split is restored
    // only when the ordered GPU pick survives reconciliation. A diffusion
    // config is sanitized to gpuMemoryMode "auto" because the mode does not apply, not because
    // the user chose Auto: writing that into the live standing preference would strand the session
    // on Auto, since the load skips saveGpuMemoryMode for diffusion and the next ordinary GGUF
    // would persist it over the user's Manual.
    gpuMemoryMode: options.isDiffusion
      ? readPersistedGpuMemoryMode()
      : (config.gpuMemoryMode ?? readPersistedGpuMemoryMode()),
    gpuLayers: config.gpuLayers ?? GPU_LAYERS_AUTO,
    nCpuMoe: config.nCpuMoe ?? 0,
    splitRatio: options.isDiffusion
      ? null
      : reconcileTensorSplit(config.tensorSplit, config.selectedGpuIds, gpuSelection.ids),
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
    engine: s.params.engine ?? "auto",
    enginePrecision: s.params.enginePrecision ?? "auto",
    engineParallelism: s.params.engineParallelism ?? "tensor",
    customContextLength: s.customContextLength ?? null,
    maxSeqLength: options.includeMaxSeqLength
      ? normalizeMaxSeqLength(s.params.maxSeqLength)
      : null,
    kvCacheDtype: s.kvCacheDtype ?? null,
    mlxKvQuant: s.mlxKvQuant ?? null,
    mlxInt8Prefill: s.mlxInt8Prefill ?? false,
    speculativeType: normalizeSpeculativeType(s.speculativeType),
    specDraftNMax: s.specDraftNMax ?? null,
    specDraftCacheDtype: s.specDraftCacheDtype ?? null,
    specDraftModel: s.specDraftModel ?? null,
    nParallel: s.nParallel ?? null,
    reasoningBudget:
      s.reasoningBudget === s.loadedReasoningBudget
        ? (s.loadedReasoningBudgetRequested ?? s.reasoningBudget)
        : s.reasoningBudget,
    reasoningBudgetMessage:
      s.reasoningBudgetMessage === s.loadedReasoningBudgetMessage
        ? (s.loadedReasoningBudgetMessageRequested ?? s.reasoningBudgetMessage)
        : s.reasoningBudgetMessage,
    nBatch: s.nBatch ?? null,
    nUbatch: s.nUbatch ?? null,
    loadMode: s.loadMode ?? null,
    ctxCheckpoints: s.ctxCheckpoints ?? null,
    cacheRam: s.cacheRam ?? null,
    tensorParallel: s.tensorParallel ?? false,
    disableVision: s.disableVision ?? false,
    chatTemplateOverride: cleanTemplate(s.chatTemplateOverride),
    // Snapshot the live GPU knobs too so a failed switch rolls the previous model's GPU Memory
    // settings back, split included.
    gpuMemoryMode: s.gpuMemoryMode,
    gpuLayers: s.gpuLayers,
    nCpuMoe: s.nCpuMoe,
    selectedGpuIds: s.selectedGpuIds,
    selectedGpuIndexKind: s.selectedGpuIndexKind,
    tensorSplit: s.splitRatio,
  };
}

/** `followGlobal` only for the running config; stored configs keep null distinct from explicit. */
export function perModelConfigsEqual(
  a: PerModelConfig,
  b: PerModelConfig,
  { followGlobal = false }: { followGlobal?: boolean } = {},
): boolean {
  const speculative = followGlobal
    ? resolvedSpeculativeType
    : normalizeSpeculativeType;
  return (
    (a.engine ?? "auto") === (b.engine ?? "auto") &&
    (a.enginePrecision ?? "auto") === (b.enginePrecision ?? "auto") &&
    (a.engineParallelism ?? "tensor") === (b.engineParallelism ?? "tensor") &&
    (a.customContextLength ?? null) === (b.customContextLength ?? null) &&
    normalizeMaxSeqLength(a.maxSeqLength) ===
      normalizeMaxSeqLength(b.maxSeqLength) &&
    (a.kvCacheDtype ?? null) === (b.kvCacheDtype ?? null) &&
    (a.mlxKvQuant ?? null) === (b.mlxKvQuant ?? null) &&
    Boolean(a.mlxInt8Prefill) === Boolean(b.mlxInt8Prefill) &&
    speculative(a.speculativeType) === speculative(b.speculativeType) &&
    (a.specDraftNMax ?? null) === (b.specDraftNMax ?? null) &&
    (a.specDraftCacheDtype ?? null) === (b.specDraftCacheDtype ?? null) &&
    (a.specDraftModel ?? null) === (b.specDraftModel ?? null) &&
    (a.nParallel ?? null) === (b.nParallel ?? null) &&
    a.reasoningBudget === b.reasoningBudget &&
    a.reasoningBudgetMessage === b.reasoningBudgetMessage &&
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

function resolvedSpeculativeType(value: string | null | undefined): string {
  return normalizeSpeculativeType(value) ?? readPersistedSpeculativeType();
}

/** Compare on the launched command, so "not loaded" and "cleared" are equal. */
function extraArgsSignature(value: string[] | null | undefined): string {
  return (value ?? []).join("\u0000");
}

function gpuFieldsEqual(a: PerModelConfig, b: PerModelConfig): boolean {
  return gpuFieldsSignature(a) === gpuFieldsSignature(b);
}
