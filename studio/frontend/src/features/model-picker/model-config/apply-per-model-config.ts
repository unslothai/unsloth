


import {
  GPU_LAYERS_AUTO,
  defaultInferenceParams,
  normalizeSpeculativeType,
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  reconcilePersistedGpuSelection,
  useChatRuntimeStore,
} from "@/features/chat";
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
    nParallel: config.nParallel ?? null,
    // the diffusion runner ignores the llama-server batch flags
    nBatch: options.isDiffusion ? null : (config.nBatch ?? null),
    nUbatch: options.isDiffusion ? null : (config.nUbatch ?? null),
    tensorParallel: options.isDiffusion
      ? false
      : (config.tensorParallel ?? false),
    chatTemplateOverride: cleanTemplate(config.chatTemplateOverride),
    // GPU Memory knobs are per-model (GGUF-only). Absent = defaults; the mode is
    // a standing preference so an absent mode falls back to the persisted one.
    // The per-GPU split ratio is never remembered, so it always resets. The GPU
    // pick is reconciled against the GPUs present now (a saved [1] on a 1-GPU
    // host would otherwise be sent and rejected).
    // A diffusion config is sanitized to gpuMemoryMode "auto" because the mode
    // does not apply to it, not because the user chose Auto. Writing that into
    // the live standing preference would strand the session on Auto: the load
    // itself skips saveGpuMemoryMode for diffusion, so nothing restores it, and
    // the next ordinary GGUF loaded without its own config sends this value and
    // persists it over the user's Manual. Keep the standing choice instead.
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
    nParallel: s.nParallel ?? null,
    nBatch: s.nBatch ?? null,
    nUbatch: s.nUbatch ?? null,
    tensorParallel: s.tensorParallel ?? false,
    chatTemplateOverride: cleanTemplate(s.chatTemplateOverride),
    // Snapshot the live GPU knobs too so a failed switch rolls the previous
    // model's GPU Memory settings back (see applyPerModelConfigToRuntime). The
    // split ratio is intentionally never remembered.
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
    (a.nParallel ?? null) === (b.nParallel ?? null) &&
    (a.nBatch ?? null) === (b.nBatch ?? null) &&
    (a.nUbatch ?? null) === (b.nUbatch ?? null) &&
    Boolean(a.tensorParallel) === Boolean(b.tensorParallel) &&
    cleanTemplate(a.chatTemplateOverride) ===
      cleanTemplate(b.chatTemplateOverride) &&
    gpuFieldsEqual(a, b)
  );
}

function gpuFieldsEqual(a: PerModelConfig, b: PerModelConfig): boolean {
  return gpuFieldsSignature(a) === gpuFieldsSignature(b);
}
