// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  applyPerModelConfigToRuntime,
  currentRuntimePerModelConfig,
  perModelConfigsEqual,
} from "@/features/model-picker/model-config/apply-per-model-config";
import {
  CONTEXT_LENGTH_MIN,
  DEFAULT_PER_MODEL_CONFIG,
  MAX_SEQ_LENGTH_MAX,
  DEFAULT_MAX_SEQ_LENGTH,
  KV_CACHE_DTYPES,
  MLX_KV_BITS,
  N_BATCH_MAX,
  N_BATCH_MIN,
  N_PARALLEL_MAX,
  N_PARALLEL_MIN,
  canonicalizeLoadMode,
  isServedByLlamaCpp,
  isServedByMlx,
  normalizeCacheRam,
  normalizeCtxCheckpoints,
  normalizeMaxSeqLength,
  savedContextPin,
  type PerModelConfig,
} from "@/features/model-picker/model-config/per-model-config";
import {
  DRAFT_N_MAX_SPEC_TYPES,
  SPECULATIVE_TYPES,
} from "@/lib/speculative-modes";
import {
  GPU_LAYERS_AUTO,
  useChatRuntimeStore,
  normalizeSpeculativeType,
} from "../stores/chat-runtime-store";
import { usePlatformStore } from "@/config/env";
import { capturedContextLength } from "./preset-policy";

/** Load/runtime knobs saved in a chat preset (excludes per-model-only blobs). */
export type PresetLoadConfig = Pick<
  PerModelConfig,
  | "customContextLength"
  | "maxSeqLength"
  | "kvCacheDtype"
  | "mlxKvBits"
  | "speculativeType"
  | "specDraftNMax"
  | "nParallel"
  | "nBatch"
  | "nUbatch"
  | "loadMode"
  | "specDraftCacheDtype"
  | "ctxCheckpoints"
  | "cacheRam"
  | "tensorParallel"
  | "disableVision"
  | "gpuMemoryMode"
  | "gpuLayers"
  | "nCpuMoe"
>;

const VALID_KV_CACHE_DTYPES = new Set<string>(KV_CACHE_DTYPES);
const VALID_SPECULATIVE_TYPES = new Set<string>(SPECULATIVE_TYPES);

export const EMPTY_PRESET_LOAD_CONFIG: PresetLoadConfig = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  mlxKvBits: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  nBatch: null,
  nUbatch: null,
  loadMode: null,
  specDraftCacheDtype: null,
  ctxCheckpoints: null,
  cacheRam: null,
  tensorParallel: false,
  disableVision: false,
};

function toComparablePerModelConfig(
  config: PresetLoadConfig,
): PerModelConfig {
  // Compared as a pin, not as whichever field the backend of the moment writes it in:
  // the same preset replayed elsewhere holds that length in the other field.
  const pin = savedContextPin(config);
  return {
    ...DEFAULT_PER_MODEL_CONFIG,
    ...config,
    customContextLength: pin,
    maxSeqLength: null,
    chatTemplateOverride: null,
    selectedGpuIds: null,
  };
}

/** A context as a preset may carry it, or null if it is not a length at all.
 *
 *  One bound for capture and for reading a saved preset back, since clamping only on the
 *  way to storage would send one window on the first replay and another after saving. The
 *  upper bound is what `/load` accepts; the lower is the control's own minimum, because a
 *  pin the control cannot represent is one the user cannot undo.
 */
function requestableContextLength(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return Math.min(MAX_SEQ_LENGTH_MAX, Math.max(CONTEXT_LENGTH_MIN, Math.floor(value)));
}

export function normalizePresetLoadConfig(
  raw: unknown,
): PresetLoadConfig | undefined {
  if (raw == null || typeof raw !== "object" || Array.isArray(raw)) {
    return undefined;
  }
  const partial = raw as Record<string, unknown>;
  const rawSpecType =
    typeof partial.speculativeType === "string"
      ? normalizeSpeculativeType(partial.speculativeType)
      : null;
  const speculativeType = rawSpecType ?? null;
  const specDraftNMax =
    speculativeType != null &&
    DRAFT_N_MAX_SPEC_TYPES.has(speculativeType) &&
    typeof partial.specDraftNMax === "number" &&
    Number.isFinite(partial.specDraftNMax)
      ? Math.max(1, Math.min(16, Math.round(partial.specDraftNMax)))
      : null;
  const gpuMemoryMode =
    partial.gpuMemoryMode === "manual" ? ("manual" as const) : undefined;
  let gpuLayers: number | undefined;
  if (typeof partial.gpuLayers === "number" && Number.isFinite(partial.gpuLayers)) {
    gpuLayers = partial.gpuLayers < 0 ? GPU_LAYERS_AUTO : Math.floor(partial.gpuLayers);
  }
  let nCpuMoe: number | undefined;
  if (typeof partial.nCpuMoe === "number" && Number.isFinite(partial.nCpuMoe)) {
    nCpuMoe = Math.max(0, Math.floor(partial.nCpuMoe));
  }

  const normalized: PresetLoadConfig = {
    customContextLength: requestableContextLength(partial.customContextLength),
    maxSeqLength: normalizeMaxSeqLength(partial.maxSeqLength as number | null),
    mlxKvBits:
      typeof partial.mlxKvBits === "number" &&
      MLX_KV_BITS.includes(partial.mlxKvBits)
        ? partial.mlxKvBits
        : null,
    kvCacheDtype:
      typeof partial.kvCacheDtype === "string" &&
      VALID_KV_CACHE_DTYPES.has(partial.kvCacheDtype)
        ? partial.kvCacheDtype
        : null,
    speculativeType:
      speculativeType && VALID_SPECULATIVE_TYPES.has(speculativeType)
        ? speculativeType
        : null,
    specDraftNMax,
    nParallel:
      typeof partial.nParallel === "number" &&
      Number.isFinite(partial.nParallel)
        ? Math.max(
            N_PARALLEL_MIN,
            Math.min(N_PARALLEL_MAX, Math.round(partial.nParallel)),
          )
        : null,
    nBatch:
      typeof partial.nBatch === "number" && Number.isFinite(partial.nBatch)
        ? Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, Math.round(partial.nBatch)))
        : null,
    nUbatch:
      typeof partial.nUbatch === "number" && Number.isFinite(partial.nUbatch)
        ? Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, Math.round(partial.nUbatch)))
        : null,
    // Through the same normalizers the per-model store uses, so a hand-edited or older preset cannot
    // smuggle in a mode or dtype the panel cannot show.
    loadMode: canonicalizeLoadMode(partial.loadMode),
    specDraftCacheDtype:
      typeof partial.specDraftCacheDtype === "string" &&
      VALID_KV_CACHE_DTYPES.has(partial.specDraftCacheDtype)
        ? partial.specDraftCacheDtype
        : null,
    ctxCheckpoints: normalizeCtxCheckpoints(partial.ctxCheckpoints),
    cacheRam: normalizeCacheRam(partial.cacheRam),
    tensorParallel:
      typeof partial.tensorParallel === "boolean"
        ? partial.tensorParallel
        : false,
    disableVision:
      typeof partial.disableVision === "boolean"
        ? partial.disableVision
        : false,
    ...(gpuMemoryMode ? { gpuMemoryMode } : {}),
    ...(gpuLayers !== undefined ? { gpuLayers } : {}),
    ...(nCpuMoe !== undefined ? { nCpuMoe } : {}),
  };

  const coalesced = coalesceDefaultLoadKnobs(normalized);
  return hasPresetLoadConfig(coalesced) ? coalesced : undefined;
}

export function hasPresetLoadConfig(
  config?: PresetLoadConfig | null,
): boolean {
  return !isSamePresetLoadConfig(config, EMPTY_PRESET_LOAD_CONFIG);
}

export function isSamePresetLoadConfig(
  a?: PresetLoadConfig | null,
  b?: PresetLoadConfig | null,
): boolean {
  return perModelConfigsEqual(
    toComparablePerModelConfig({ ...EMPTY_PRESET_LOAD_CONFIG, ...a }),
    toComparablePerModelConfig({ ...EMPTY_PRESET_LOAD_CONFIG, ...b }),
  );
}

export function capturePresetLoadConfig(): PresetLoadConfig | undefined {
  const snapshot = currentRuntimePerModelConfig({ includeMaxSeqLength: true });
  const store = useChatRuntimeStore.getState();
  const isGguf = isServedByLlamaCpp({
    loadedIsGguf: store.loadedIsGguf,
    activeGgufVariant: store.activeGgufVariant,
    activeNativePathToken: store.activeNativePathToken,
    checkpoint: store.params.checkpoint,
  });
  const platform = usePlatformStore.getState();
  const isMlx = isServedByMlx(isGguf, platform.deviceType, platform.chatOnlyReason);
  // The same bound a saved preset is read back under; this one replays from memory first.
  const effectiveContextLength = requestableContextLength(
    capturedContextLength({
      isGguf,
      controlPin: snapshot.customContextLength,
      loadedContextLength: store.loadedContextLength,
    }),
  );
  const captured: PresetLoadConfig = {
    customContextLength: effectiveContextLength ?? null,
    maxSeqLength: isMlx ? null : normalizeMaxSeqLength(snapshot.maxSeqLength),
    kvCacheDtype: snapshot.kvCacheDtype ?? null,
    mlxKvBits: snapshot.mlxKvBits ?? null,
    speculativeType: normalizeSpeculativeType(snapshot.speculativeType),
    specDraftNMax: snapshot.specDraftNMax ?? null,
    nParallel: snapshot.nParallel ?? null,
    nBatch: snapshot.nBatch ?? null,
    nUbatch: snapshot.nUbatch ?? null,
    loadMode: snapshot.loadMode ?? null,
    specDraftCacheDtype: snapshot.specDraftCacheDtype ?? null,
    ctxCheckpoints: snapshot.ctxCheckpoints ?? null,
    cacheRam: snapshot.cacheRam ?? null,
    tensorParallel: snapshot.tensorParallel ?? false,
    disableVision: snapshot.disableVision ?? false,
    ...(snapshot.gpuMemoryMode === "manual"
      ? { gpuMemoryMode: "manual" as const }
      : {}),
    ...(snapshot.gpuLayers != null && snapshot.gpuLayers >= 0
      ? { gpuLayers: snapshot.gpuLayers }
      : snapshot.gpuMemoryMode === "manual"
        ? { gpuLayers: GPU_LAYERS_AUTO }
        : {}),
    ...(snapshot.nCpuMoe != null && snapshot.nCpuMoe > 0
      ? { nCpuMoe: snapshot.nCpuMoe }
      : {}),
  };
  const coalesced = coalesceDefaultLoadKnobs(captured);
  return hasPresetLoadConfig(coalesced) ? coalesced : undefined;
}

function coalesceDefaultLoadKnobs(
  captured: PresetLoadConfig,
): PresetLoadConfig {
  const result: PresetLoadConfig = { ...captured };
  if (normalizeMaxSeqLength(result.maxSeqLength) === DEFAULT_MAX_SEQ_LENGTH) {
    result.maxSeqLength = null;
  }
  const speculativeType = normalizeSpeculativeType(result.speculativeType);
  if (speculativeType == null || speculativeType === "auto") {
    result.speculativeType = null;
  }
  if (
    (result.gpuLayers == null || result.gpuLayers < 0) &&
    result.gpuMemoryMode !== "manual"
  ) {
    delete result.gpuLayers;
  }
  if ((result.nCpuMoe ?? 0) === 0) {
    delete result.nCpuMoe;
  }
  return result;
}

export function applyPresetLoadConfig(
  config?: PresetLoadConfig | null,
): void {
  if (config == null) {
    return;
  }
  const store = useChatRuntimeStore.getState();
  applyPerModelConfigToRuntime({
    ...DEFAULT_PER_MODEL_CONFIG,
    maxSeqLength: normalizeMaxSeqLength(config.maxSeqLength) ?? DEFAULT_MAX_SEQ_LENGTH,
    customContextLength: config.customContextLength ?? null,
    kvCacheDtype: config.kvCacheDtype ?? null,
    mlxKvBits: config.mlxKvBits ?? null,
    speculativeType: config.speculativeType ?? null,
    specDraftNMax: config.specDraftNMax ?? null,
    nParallel: config.nParallel ?? null,
    nBatch: config.nBatch ?? null,
    nUbatch: config.nUbatch ?? null,
    loadMode: config.loadMode ?? null,
    specDraftCacheDtype: config.specDraftCacheDtype ?? null,
    ctxCheckpoints: config.ctxCheckpoints ?? null,
    cacheRam: config.cacheRam ?? null,
    tensorParallel: config.tensorParallel ?? false,
    disableVision: config.disableVision ?? false,
    chatTemplateOverride: null,
    gpuMemoryMode: config.gpuMemoryMode,
    gpuLayers: config.gpuLayers,
    nCpuMoe: config.nCpuMoe,
    selectedGpuIds: store.selectedGpuIds,
    selectedGpuIndexKind: store.selectedGpuIndexKind,
  });
}

export function formatPresetLoadConfigSummary(
  config?: PresetLoadConfig | null,
): string | null {
  if (!config || !hasPresetLoadConfig(config)) {
    return null;
  }
  const parts: string[] = [];
  if (config.customContextLength != null) {
    parts.push(`Ctx ${config.customContextLength.toLocaleString()}`);
  }
  if (config.kvCacheDtype) {
    parts.push(`KV ${config.kvCacheDtype}`);
  }
  if (config.mlxKvBits) {
    parts.push(`MLX KV ${config.mlxKvBits}-bit`);
  }
  if (config.speculativeType && config.speculativeType !== "auto") {
    parts.push(`Spec ${config.speculativeType}`);
  }
  if (config.nParallel != null) {
    parts.push(`${config.nParallel} slots`);
  }
  if (config.nBatch != null) {
    parts.push(`Batch ${config.nBatch}`);
  }
  if (config.nUbatch != null) {
    parts.push(`uBatch ${config.nUbatch}`);
  }
  if (config.loadMode) {
    parts.push(`Load ${config.loadMode}`);
  }
  if (config.specDraftCacheDtype) {
    parts.push(`Draft KV ${config.specDraftCacheDtype}`);
  }
  if (config.ctxCheckpoints != null) {
    parts.push(`${config.ctxCheckpoints} checkpoints`);
  }
  if (config.cacheRam != null) {
    parts.push(`Cache RAM ${config.cacheRam}`);
  }
  if (config.gpuMemoryMode === "manual") {
    parts.push("GPU manual");
  }
  if (config.gpuLayers != null && config.gpuLayers >= 0) {
    parts.push(`${config.gpuLayers} layers`);
  }
  if (config.tensorParallel) {
    parts.push("TP");
  }
  if (config.disableVision) {
    parts.push("No vision");
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}
