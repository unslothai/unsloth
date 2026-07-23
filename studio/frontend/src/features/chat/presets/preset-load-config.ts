// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  applyPerModelConfigToRuntime,
  currentRuntimePerModelConfig,
  perModelConfigsEqual,
} from "@/features/model-picker";
import {
  CONTEXT_LENGTH_MIN,
  DEFAULT_PER_MODEL_CONFIG,
  DEFAULT_MAX_SEQ_LENGTH,
  KV_CACHE_DTYPES,
  MTP_SPECULATIVE_TYPES,
  SPECULATIVE_TYPES,
  normalizeMaxSeqLength,
  type PerModelConfig,
} from "@/features/model-picker/model-config/per-model-config";
import { GPU_LAYERS_AUTO, normalizeSpeculativeType } from "../stores/chat-runtime-store";

/** Load/runtime knobs saved in a chat preset (excludes per-model-only blobs). */
export type PresetLoadConfig = Pick<
  PerModelConfig,
  | "customContextLength"
  | "maxSeqLength"
  | "kvCacheDtype"
  | "speculativeType"
  | "specDraftNMax"
  | "tensorParallel"
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
  speculativeType: null,
  specDraftNMax: null,
  tensorParallel: false,
};

function toComparablePerModelConfig(
  config: PresetLoadConfig,
): PerModelConfig {
  return {
    ...DEFAULT_PER_MODEL_CONFIG,
    ...config,
    chatTemplateOverride: null,
    selectedGpuIds: null,
  };
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
    MTP_SPECULATIVE_TYPES.has(speculativeType) &&
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
    customContextLength:
      typeof partial.customContextLength === "number" &&
      Number.isFinite(partial.customContextLength) &&
      partial.customContextLength > 0
        ? Math.max(CONTEXT_LENGTH_MIN, Math.floor(partial.customContextLength))
        : null,
    maxSeqLength: normalizeMaxSeqLength(partial.maxSeqLength as number | null),
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
    tensorParallel:
      typeof partial.tensorParallel === "boolean"
        ? partial.tensorParallel
        : false,
    ...(gpuMemoryMode ? { gpuMemoryMode } : {}),
    ...(gpuLayers !== undefined ? { gpuLayers } : {}),
    ...(nCpuMoe !== undefined ? { nCpuMoe } : {}),
  };

  return hasPresetLoadConfig(normalized) ? normalized : undefined;
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
  const captured: PresetLoadConfig = {
    customContextLength: snapshot.customContextLength ?? null,
    maxSeqLength: normalizeMaxSeqLength(snapshot.maxSeqLength),
    kvCacheDtype: snapshot.kvCacheDtype ?? null,
    speculativeType: normalizeSpeculativeType(snapshot.speculativeType),
    specDraftNMax: snapshot.specDraftNMax ?? null,
    tensorParallel: snapshot.tensorParallel ?? false,
    ...(snapshot.gpuMemoryMode === "manual"
      ? { gpuMemoryMode: "manual" as const }
      : {}),
    ...(snapshot.gpuLayers != null && snapshot.gpuLayers >= 0
      ? { gpuLayers: snapshot.gpuLayers }
      : snapshot.gpuLayers != null && snapshot.gpuLayers < 0
        ? { gpuLayers: GPU_LAYERS_AUTO }
        : {}),
    ...(snapshot.nCpuMoe != null && snapshot.nCpuMoe > 0
      ? { nCpuMoe: snapshot.nCpuMoe }
      : {}),
  };
  return hasPresetLoadConfig(captured) ? captured : undefined;
}

export function applyPresetLoadConfig(
  config?: PresetLoadConfig | null,
): void {
  applyPerModelConfigToRuntime({
    ...DEFAULT_PER_MODEL_CONFIG,
    maxSeqLength: normalizeMaxSeqLength(config?.maxSeqLength) ?? DEFAULT_MAX_SEQ_LENGTH,
    customContextLength: config?.customContextLength ?? null,
    kvCacheDtype: config?.kvCacheDtype ?? null,
    speculativeType: config?.speculativeType ?? null,
    specDraftNMax: config?.specDraftNMax ?? null,
    tensorParallel: config?.tensorParallel ?? false,
    chatTemplateOverride: null,
    gpuMemoryMode: config?.gpuMemoryMode,
    gpuLayers: config?.gpuLayers,
    nCpuMoe: config?.nCpuMoe,
    selectedGpuIds: null,
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
  if (config.speculativeType && config.speculativeType !== "auto") {
    parts.push(`Spec ${config.speculativeType}`);
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
  return parts.length > 0 ? parts.join(" · ") : null;
}
