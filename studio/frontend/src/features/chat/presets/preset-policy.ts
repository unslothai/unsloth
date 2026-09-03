// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "../types/runtime";
import type { PresetLoadConfig } from "./preset-load-config";

export const defaultInferenceParams = DEFAULT_INFERENCE_PARAMS;

/** The fewest tokens the Max Tokens control offers, and so the least any ceiling may be. */
export const MAX_TOKENS_MIN = 64;

export interface Preset {
  name: string;
  params: InferenceParams;
  /** Optional GGUF/load knobs captured with the preset. */
  loadConfig?: PresetLoadConfig;
}

export type PresetOwnedParams = Pick<
  InferenceParams,
  | "temperature"
  | "topP"
  | "topK"
  | "minP"
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
    repetitionPenalty: params.repetitionPenalty,
    presencePenalty: params.presencePenalty,
    maxTokens: params.maxTokens,
    systemPrompt: params.systemPrompt ?? "",
    systemVariables: params.systemVariables ?? "",
    // Normalised, so a preset saved before the field existed never reads as modified.
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
  /** The window the response reports, as the context constructor reads it -- not the raw
   *  field, where a backend that sizes nothing echoes the length it was asked for. */
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

  // An unreported window leaves Max Tokens on the settings sheet's ceiling, not on
  // whatever the previous model left behind.
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

/** The user's own context pin behind a load request, or null where the request is not one.
 *
 *  A record written before the MLX pin moved fields carries it in `maxSeqLength`, which is
 *  the value `resolveLoadMaxSeqLength` builds the request from, so the completed load has
 *  to pin the same number or the UI reads a pinned runtime as Auto. llama.cpp's
 *  `maxSeqLength` is not a context pin, so only MLX admits it.
 */
export function loadRequestContextPin(
  customContextLength: number | null,
  isMlx: boolean | null | undefined,
  pinnedMaxSeqLength: number | null,
): number | null {
  return customContextLength ?? (isMlx ? pinnedMaxSeqLength : null);
}

/** The context pin to keep for a model that has just loaded, or null if it has none.
 *
 *  They pin for different reasons: llama.cpp's matters only under manual memory with
 *  auto layers, where `--fit` owns sizing; MLX sizes itself when asked for nothing, so
 *  any positive request was the user's choice.
 */
export function retainedContextPin({
  isMlx,
  requestedContextLength,
}: {
  isMlx?: boolean | null;
  requestedContextLength: number | null;
}): number | null {
  return isMlx && (requestedContextLength ?? 0) > 0 ? requestedContextLength : null;
}

/** The context a preset records, in the one field that replays as a pin.
 *
 *  A window nobody pinned is not recorded: replaying asks for nothing and the backend
 *  arrives there again, while storing it would turn it into a request. llama.cpp is the
 *  exception, since its window depends on the machine.
 */
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

/** The context length to record for a model that has just loaded.
 *
 *  The reported window, not the request: a self-sizing backend was asked with the
 *  non-positive sentinel. Only where nothing was reported does the request stand.
 */
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

/** What bounds Max Tokens when the model reported no window at all.
 *
 *  llama.cpp reports one whenever it can read it, so a missing one is a failed read and
 *  the held value stands. A backend that never reports keeps the app default, since the
 *  session's length is a request and on a model change is the outgoing model's.
 */
export function unreportedWindowMaxTokens(
  isGguf: boolean,
  currentMaxTokens: number,
): number {
  return isGguf ? currentMaxTokens : defaultInferenceParams.maxSeqLength;
}

/** The most tokens a local model may be asked to generate.
 *
 *  The control's minimum outranks the window: a slider whose maximum is below its
 *  minimum cannot be operated. Reachable, since MLX honours a tiny positive request.
 */
export function localMaxTokensCeiling(
  loadedContextLength: number | null,
  unreportedWindowFallback: number,
): number {
  return Math.max(MAX_TOKENS_MIN, loadedContextLength ?? unreportedWindowFallback);
}

/** The cap a load hands `getReplayedParams`, floored the same way the ceiling above is.
 *
 *  That clamp only lowers Max Tokens, so a raw window below the control's minimum would
 *  leave the value outside its own slider. `undefined` means nothing sized a window.
 */
export function replayMaxTokensCap(
  loadedContextLength: number | null | undefined,
): number | undefined {
  return loadedContextLength == null
    ? undefined
    : Math.max(MAX_TOKENS_MIN, loadedContextLength);
}

/** The app-level request to fall back on when the target has no pin of its own.
 *
 *  `params.maxSeqLength` holds a REQUEST only while the backend serving it does not size
 *  its own window. A load that sized one wrote its RESOLVED window there instead, so
 *  carrying that value into the next model asks transformers to allocate the outgoing
 *  model's window: leaving a 131072 MLX model for an unconfigured one loads it at 131072.
 */
export function unpinnedDefaultRequest(
  outgoingSizedItsOwnWindow: boolean | null | undefined,
  sessionMaxSeqLength: number | null | undefined,
  appDefault: number,
): number {
  if (outgoingSizedItsOwnWindow) return appDefault;
  return sessionMaxSeqLength || appDefault;
}

/** What a load asks for when the user has pinned no context of their own.
 *
 *  Both local backends take the same non-positive sentinel to mean "size it yourself".
 *  The app default covers the remaining case, where nothing fits a window.
 */
export function unpinnedLoadContext(
  isGgufLoad: boolean,
  isMlx: boolean | null | undefined,
  appDefault: number,
): number {
  return isGgufLoad || isMlx ? 0 : appDefault;
}

/**
 * Adjust a resolved max-seq-length for the GPU Memory mode. Under Manual + Auto
 * layers (GGUF, gpuLayers < 0) llama.cpp's --fit owns context sizing, so send 0
 * (the backend omits -c) unless the user pinned a length; every other case keeps
 * the resolved fallback. Shared by every GGUF load path so they can't drift.
 */
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

/**
 * The context pin a completed load leaves behind: the Context Length the user
 * EXPLICITLY set for it, or null for Auto.
 *
 * Takes the user's setting, never the n_ctx that went on the wire. The wire
 * value cannot answer this: `resolveLoadMaxSeqLength`'s isReloadingCurrentGguf
 * branch sends the resolved context on a same-model reload so the reload does
 * not resize, so under a custom or modified preset an Auto load puts a positive
 * n_ctx on the wire too. Reading a pin back out of that turned Auto into a
 * numeric pin at the current context on any same-model reload, after which a GPU
 * memory change could no longer auto-resize.
 *
 * Takes no GPU Memory mode, deliberately. `resolveLoadMaxSeqLength` honors a pin
 * in every mode, but the predicate this replaces kept one only under Manual +
 * Auto layers: a survival rule narrower than the send rule, so a load on Default
 * sent the user's pin and then dropped it on completion. That was the bug.
 *
 * Callers keep their own isGguf/targetIsGguf guard inline: a non-GGUF load has
 * no n_ctx, and its max_seq_length must not be pinned as one.
 */
export function resolveExplicitCtxPin(
  customContextLength: number | null | undefined,
): number | null {
  return customContextLength && customContextLength > 0
    ? customContextLength
    : null;
}
