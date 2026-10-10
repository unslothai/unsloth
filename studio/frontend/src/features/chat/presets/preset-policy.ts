// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "../types/runtime";
import { effectiveMinPMode } from "../lib/min-p-policy.ts";
import type { PresetLoadConfig } from "./preset-load-config";

export const defaultInferenceParams = DEFAULT_INFERENCE_PARAMS;

export const MAX_TOKENS_MIN = 64;

export interface Preset {
  name: string;
  params: InferenceParams;
  loadConfig?: PresetLoadConfig;
}

export type PresetOwnedParams = Pick<
  InferenceParams,
  | "temperature"
  | "topP"
  | "topK"
  | "minP"
  | "minPMode"
  | "repetitionPenalty"
  | "presencePenalty"
  | "maxTokens"
  | "systemPrompt"
  | "systemVariables"
  | "seed"
>;

export const BUILTIN_PRESETS: Preset[] = [
  { name: "Default", params: { ...defaultInferenceParams } },
];

export const BUILTIN_PRESET_NAMES = new Set(
  BUILTIN_PRESETS.map((preset) => preset.name),
);

export type ChatPresetSource = "builtin-default" | "custom" | "modified";

export function getPresetSource(name: string): ChatPresetSource {
  return name === "Default" ? "builtin-default" : "custom";
}

export function getUniquePresetName(
  baseName: string,
  usedNames: Set<string>,
): string {
  const normalizedBase = baseName.trim() || "Imported Prompt";
  let nextName = normalizedBase;
  let suffix = 2;
  while (usedNames.has(nextName)) {
    nextName = `${normalizedBase} ${suffix}`;
    suffix += 1;
  }
  usedNames.add(nextName);
  return nextName;
}

export function getBuiltinVariantName(
  baseName: string,
  usedNames: Set<string>,
): string {
  const normalizedBase = baseName.trim() || "Imported Prompt";
  let suffix = 1;
  let nextName = `${normalizedBase} ${suffix}`;
  while (usedNames.has(nextName)) {
    suffix += 1;
    nextName = `${normalizedBase} ${suffix}`;
  }
  usedNames.add(nextName);
  return nextName;
}

export function normalizeCustomPresets(presets: Preset[]): Preset[] {
  const usedNames = new Set(BUILTIN_PRESET_NAMES);
  return presets
    .map((preset): Preset | null => {
      const trimmedName = preset.name.trim();
      if (!trimmedName) {
        return null;
      }
      const name = usedNames.has(trimmedName)
        ? getBuiltinVariantName(trimmedName, usedNames)
        : trimmedName;
      usedNames.add(name);
      return {
        name,
        params: preset.params,
        ...(preset.loadConfig ? { loadConfig: preset.loadConfig } : {}),
      };
    })
    .filter((preset): preset is Preset => preset !== null);
}

export function getOrderedPresets(customPresets: Preset[]): Preset[] {
  return [...BUILTIN_PRESETS, ...normalizeCustomPresets(customPresets)];
}

export function getPresetOwnedParams(
  params: InferenceParams,
): PresetOwnedParams {
  return {
    temperature: params.temperature,
    topP: params.topP,
    topK: params.topK,
    minP: params.minP,
    minPMode: effectiveMinPMode(params),
    repetitionPenalty: params.repetitionPenalty,
    presencePenalty: params.presencePenalty,
    maxTokens: params.maxTokens,
    systemPrompt: params.systemPrompt ?? "",
    systemVariables: params.systemVariables ?? "",
    // Normalised so presets saved before the field existed are not modified.
    seed: params.seed ?? null,
  };
}

export function isSamePresetConfig(
  a: InferenceParams,
  b: InferenceParams,
): boolean {
  const left = getPresetOwnedParams(a);
  const right = getPresetOwnedParams(b);
  return (
    left.temperature === right.temperature &&
    left.topP === right.topP &&
    left.topK === right.topK &&
    left.minP === right.minP &&
    left.minPMode === right.minPMode &&
    left.repetitionPenalty === right.repetitionPenalty &&
    left.presencePenalty === right.presencePenalty &&
    left.maxTokens === right.maxTokens &&
    left.systemPrompt === right.systemPrompt &&
    left.systemVariables === right.systemVariables &&
    left.seed === right.seed
  );
}

export function getPresetOwnedConfigKey(params: InferenceParams): string {
  return JSON.stringify(getPresetOwnedParams(params));
}

export function toPresetParams(params: InferenceParams): InferenceParams {
  return {
    ...defaultInferenceParams,
    ...getPresetOwnedParams(params),
  };
}

export function applyPresetParams(
  current: InferenceParams,
  preset: InferenceParams,
): InferenceParams {
  return {
    ...current,
    ...getPresetOwnedParams(preset),
  };
}

export function applyPresetForProvider(
  current: InferenceParams,
  preset: Preset,
  providerType: string | null | undefined,
): InferenceParams {
  const applied = applyPresetParams(current, preset.params);
  if (preset.name !== "Default") return applied;
  return {
    ...applied,
    minPMode: providerType === "vllm" ? "server-default" : effectiveMinPMode(current),
  };
}

export type PresetSaveMode =
  | "disabled"
  | "overwrite-active"
  | "overwrite-other"
  | "copy-builtin"
  | "create";

export interface PresetSaveState {
  mode: PresetSaveMode;
  canSubmit: boolean;
  isSaveReady: boolean;
  buttonLabel: string;
  title: string;
}

export function getPresetSaveState({
  rawName,
  activePreset,
  presets,
  hasUnsavedPresetChanges,
}: {
  rawName: string;
  activePreset: string;
  presets: Preset[];
  hasUnsavedPresetChanges: boolean;
}): PresetSaveState {
  const trimmedName = rawName.trim();
  if (!trimmedName) {
    return {
      mode: "disabled",
      canSubmit: false,
      isSaveReady: false,
      buttonLabel: "Save",
      title: "Enter a preset name",
    };
  }

  if (BUILTIN_PRESET_NAMES.has(trimmedName)) {
    const variantName = getBuiltinVariantName(
      trimmedName,
      new Set(presets.map((preset) => preset.name)),
    );
    return {
      mode: "copy-builtin",
      canSubmit: activePreset !== trimmedName || hasUnsavedPresetChanges,
      isSaveReady: activePreset !== trimmedName || hasUnsavedPresetChanges,
      buttonLabel:
        activePreset === trimmedName && !hasUnsavedPresetChanges
          ? "Saved"
          : "Save",
      title:
        activePreset === trimmedName && !hasUnsavedPresetChanges
          ? "No unsaved changes"
          : `Save current settings as "${variantName}"`,
    };
  }

  const matchingPreset = presets.find((preset) => preset.name === trimmedName);
  if (matchingPreset) {
    const isActiveMatch = matchingPreset.name === activePreset;
    return {
      mode: isActiveMatch ? "overwrite-active" : "overwrite-other",
      canSubmit: !isActiveMatch || hasUnsavedPresetChanges,
      isSaveReady: !isActiveMatch || hasUnsavedPresetChanges,
      buttonLabel: isActiveMatch && !hasUnsavedPresetChanges ? "Saved" : "Save",
      title: isActiveMatch
        ? hasUnsavedPresetChanges
          ? "Save current settings to this preset"
          : "No unsaved changes"
        : `Overwrite preset "${trimmedName}"`,
    };
  }

  return {
    mode: "create",
    canSubmit: true,
    isSaveReady: true,
    buttonLabel: "Save",
    title: `Save current settings as "${trimmedName}"`,
  };
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return value;
}

interface BackendInferenceDefaults {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  trust_remote_code?: boolean;
}

export interface BackendInferenceEnvelope {
  is_gguf?: boolean;
  context_length?: number | null;
  inference?: BackendInferenceDefaults | null;
}

export function mergeBackendRecommendedInference({
  current,
  response,
  modelId,
  presetSource,
  loadedContextLength,
}: {
  current: InferenceParams;
  response: BackendInferenceEnvelope;
  modelId: string;
  presetSource: ChatPresetSource;
  /** The constructor's reading, not the raw field, which may echo the request. */
  loadedContextLength: number | null;
}): InferenceParams {
  const inference = response.inference;
  const next: InferenceParams = {
    ...current,
    checkpoint: modelId,
    trustRemoteCode:
      typeof inference?.trust_remote_code === "boolean"
        ? inference.trust_remote_code
        : current.trustRemoteCode,
  };

  if (presetSource !== "builtin-default") {
    return next;
  }

  const defaultMaxTokens = localMaxTokensCeiling(
    loadedContextLength,
    unreportedWindowMaxTokens(response.is_gguf ?? false, current.maxTokens),
  );
  return {
    ...next,
    maxTokens: defaultMaxTokens,
    temperature:
      toFiniteNumber(inference?.temperature) ??
      defaultInferenceParams.temperature,
    topP: toFiniteNumber(inference?.top_p) ?? defaultInferenceParams.topP,
    topK: toFiniteNumber(inference?.top_k) ?? defaultInferenceParams.topK,
    minP: toFiniteNumber(inference?.min_p) ?? defaultInferenceParams.minP,
    presencePenalty:
      toFiniteNumber(inference?.presence_penalty) ??
      defaultInferenceParams.presencePenalty,
  };
}

export function resolveLoadMaxSeqLength({
  modelId,
  ggufVariant,
  isGguf,
  customContextLength,
  loadedContextLength,
  currentCheckpoint,
  activeGgufVariant,
  isMlx,
  pinnedMaxSeqLength,
  defaultMaxSeqLength,
  presetSource,
}: {
  modelId: string;
  ggufVariant?: string | null;
  isGguf?: boolean | null;
  customContextLength: number | null;
  loadedContextLength: number | null;
  currentCheckpoint: string;
  activeGgufVariant?: string | null;
  isMlx?: boolean | null;
  pinnedMaxSeqLength: number | null;
  defaultMaxSeqLength: number;
  presetSource: ChatPresetSource;
}): number {
  const isDirectGgufFile = modelId.toLowerCase().endsWith(".gguf");
  const isGgufLoad = isGguf === true || ggufVariant != null || isDirectGgufFile;
  const isReloadingCurrentGguf =
    isGgufLoad &&
    currentCheckpoint === modelId &&
    (ggufVariant ?? null) === (activeGgufVariant ?? null);

  if (customContextLength != null) {
    return customContextLength;
  }
  if (isGgufLoad && presetSource === "builtin-default") {
    return 0;
  }
  if (isReloadingCurrentGguf) {
    return loadedContextLength ?? 0;
  }
  if (isGgufLoad) {
    return 0;
  }
  if (pinnedMaxSeqLength != null) {
    return pinnedMaxSeqLength;
  }
  return unpinnedLoadContext(isGgufLoad, isMlx, defaultMaxSeqLength);
}

/** Older MLX records keep the pin in `maxSeqLength`; llama.cpp's is not a pin. */
export function loadRequestContextPin(
  customContextLength: number | null,
  isMlx: boolean | null | undefined,
  pinnedMaxSeqLength: number | null,
): number | null {
  return customContextLength ?? (isMlx ? pinnedMaxSeqLength : null);
}

/** llama.cpp pins only under manual memory + auto layers; any positive MLX request is a pin. */
export function retainedContextPin({
  isMlx,
  requestedContextLength,
}: {
  isMlx?: boolean | null;
  requestedContextLength: number | null;
}): number | null {
  return isMlx && (requestedContextLength ?? 0) > 0 ? requestedContextLength : null;
}

/** Unpinned windows are not recorded, except llama.cpp whose window depends on the machine. */
export function capturedContextLength({
  isGguf,
  controlPin,
  loadedContextLength,
}: {
  isGguf: boolean;
  controlPin: number | null | undefined;
  loadedContextLength: number | null | undefined;
}): number | null {
  return controlPin ?? (isGguf ? (loadedContextLength ?? null) : null);
}

/** The reported window, not the request, which may be a sentinel. */
export function loadedContextForParams(
  reportedContextLength: number | null | undefined,
  requestedMaxSeqLength: number,
  previousMaxSeqLength: number,
): number {
  if (reportedContextLength != null) {
    return reportedContextLength;
  }
  return requestedMaxSeqLength > 0 ? requestedMaxSeqLength : previousMaxSeqLength;
}

export function unreportedWindowMaxTokens(
  isGguf: boolean,
  currentMaxTokens: number,
): number {
  return isGguf ? currentMaxTokens : defaultInferenceParams.maxSeqLength;
}

/** The control's minimum outranks the window so the slider stays operable. */
export function localMaxTokensCeiling(
  loadedContextLength: number | null,
  unreportedWindowFallback: number,
): number {
  return Math.max(MAX_TOKENS_MIN, loadedContextLength ?? unreportedWindowFallback);
}

export function replayMaxTokensCap(
  loadedContextLength: number | null | undefined,
): number | undefined {
  return loadedContextLength == null
    ? undefined
    : Math.max(MAX_TOKENS_MIN, loadedContextLength);
}

/** `maxSeqLength` may hold a resolved window, which must not carry into the next model. */
export function unpinnedDefaultRequest(
  outgoingSizedItsOwnWindow: boolean | null | undefined,
  sessionMaxSeqLength: number | null | undefined,
  appDefault: number,
): number {
  if (outgoingSizedItsOwnWindow) return appDefault;
  return sessionMaxSeqLength || appDefault;
}

export function unpinnedLoadContext(
  isGgufLoad: boolean,
  isMlx: boolean | null | undefined,
  appDefault: number,
): number {
  return isGgufLoad || isMlx ? 0 : appDefault;
}

/** Under Manual + Auto layers, --fit owns context, so send 0 unless the user pinned one. */
export function resolveFitMaxSeqLength(
  isGguf: boolean | null | undefined,
  gpuMemoryMode: "auto" | "manual",
  gpuLayers: number,
  customContextLength: number | null,
  fallback: number,
): number {
  if (!isGguf || gpuMemoryMode !== "manual" || gpuLayers >= 0) return fallback;
  return customContextLength && customContextLength > 0 ? customContextLength : 0;
}

export function isReplayedLoadContext(
  isGguf: boolean | null | undefined,
  customContextLength: number | null,
  maxSeqLength: number,
): boolean {
  return isGguf === true && customContextLength == null && maxSeqLength > 0;
}

/** The user's explicit Context Length or null; never derived from the wire n_ctx. */
export function resolveExplicitCtxPin(
  customContextLength: number | null | undefined,
): number | null {
  return customContextLength && customContextLength > 0
    ? customContextLength
    : null;
}
