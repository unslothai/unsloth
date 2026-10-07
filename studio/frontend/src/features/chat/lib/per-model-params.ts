// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-checkpoint param memory. No store or network imports, so it stays unit-testable.

import type {
  InferenceParams,
  PersistedInferenceParams,
} from "../types/runtime";

import { normalizeSavedMinP } from "./min-p-policy.ts";

export type PersistedInferenceParamKey = keyof PersistedInferenceParams;

export const PERSISTED_INFERENCE_PARAM_KEYS = [
  "temperature",
  "topP",
  "topK",
  "minP",
  "minPMode",
  "repetitionPenalty",
  "presencePenalty",
  "maxSeqLength",
  "maxTokens",
  "systemPrompt",
  "systemVariables",
  "trustRemoteCode",
  "fastMode",
  "seed",
] as const satisfies readonly PersistedInferenceParamKey[];

/** maxSeqLength is excluded: context belongs to the load config. */
export const REMEMBERED_INFERENCE_PARAM_KEYS =
  PERSISTED_INFERENCE_PARAM_KEYS.filter(
    (key): key is PersistedInferenceParamKey => key !== "maxSeqLength",
  );

export function setInferenceParam(
  params: InferenceParams,
  key: PersistedInferenceParamKey,
  value: PersistedInferenceParams[PersistedInferenceParamKey],
): void {
  (params as Record<PersistedInferenceParamKey, unknown>)[key] = value;
}

export function pickRememberedParams(
  params: InferenceParams,
): PersistedInferenceParams {
  const picked: PersistedInferenceParams = {};
  for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
    const value = params[key];
    if (value !== undefined) {
      setInferenceParam(picked as InferenceParams, key, value);
    }
  }
  return picked;
}

function movedRememberedParam(changed: PersistedInferenceParams): boolean {
  return REMEMBERED_INFERENCE_PARAM_KEYS.some(
    (key) => changed[key] !== undefined,
  );
}

/** Only moved keys: the server merges per key, so a full snapshot would clobber other tabs. */
export function pickRememberedChanges(
  changed: PersistedInferenceParams,
): PersistedInferenceParams {
  const picked: PersistedInferenceParams = {};
  for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
    const value = changed[key];
    if (value !== undefined) {
      setInferenceParam(picked as InferenceParams, key, value);
    }
  }
  return picked;
}

export function getRememberedParamsPatch(
  enabled: boolean,
  paramsByModel: Record<string, PersistedInferenceParams>,
  modelId: string | undefined,
  changedParams: PersistedInferenceParams,
  snapshot: PersistedInferenceParams,
): Record<string, PersistedInferenceParams> | null {
  if (!(enabled && modelId && movedRememberedParam(changedParams))) {
    return null;
  }
  // Whole snapshot: replay overlays it, so a partial one leaves stale gaps.
  return { ...paramsByModel, [modelId]: snapshot };
}

/** Returns `current` by identity when nothing is remembered. */
export function getReplayedParams(
  enabled: boolean,
  paramsByModel: Record<string, PersistedInferenceParams>,
  current: InferenceParams,
  modelId: string,
  checkpointChanged: boolean,
  maxTokensCap?: number,
): InferenceParams {
  const capped = (params: InferenceParams): InferenceParams =>
    maxTokensCap !== undefined && params.maxTokens > maxTokensCap
      ? { ...params, maxTokens: maxTokensCap }
      : params;
  if (!(enabled && checkpointChanged)) {
    return capped(current);
  }
  const saved = paramsByModel[modelId];
  const remembered = saved ? normalizeSavedMinP(saved) : undefined;
  if (!remembered) {
    return capped(current);
  }
  // Key by key, not a spread, so maxSeqLength and stale keys do not leak into params.
  const replayed = { ...current };
  for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
    const value = remembered[key];
    if (value !== undefined) {
      setInferenceParam(replayed, key, value);
    }
  }
  if (maxTokensCap !== undefined && replayed.maxTokens > maxTokensCap) {
    replayed.maxTokens = maxTokensCap;
  }
  return replayed;
}
