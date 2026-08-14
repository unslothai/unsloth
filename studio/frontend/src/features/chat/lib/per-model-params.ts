// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-model parameter memory. Chat settings hold one set of sampling params, so
// switching models used to hand the next model the previous one's temperature and
// prompt. These keep a per-checkpoint record alongside the global set: an edit is
// filed against the model it was made for, and a switch replays that model's own
// settings. No store or network imports, so the rules stay unit-testable.

import type {
  InferenceParams,
  PersistedInferenceParams,
} from "../types/runtime";

export type PersistedInferenceParamKey = keyof PersistedInferenceParams;

/** Params that persist across a reload. `checkpoint` names the model, so it is
 * the key rather than one of the values. */
export const PERSISTED_INFERENCE_PARAM_KEYS = [
  "temperature",
  "topP",
  "topK",
  "minP",
  "repetitionPenalty",
  "presencePenalty",
  "maxSeqLength",
  "maxTokens",
  "systemPrompt",
  "systemVariables",
  "trustRemoteCode",
  "fastMode",
] as const satisfies readonly PersistedInferenceParamKey[];

export function setInferenceParam(
  params: InferenceParams,
  key: PersistedInferenceParamKey,
  value: PersistedInferenceParams[PersistedInferenceParamKey],
): void {
  (params as Record<PersistedInferenceParamKey, unknown>)[key] = value;
}

/** The persisted subset of a params object. No version-bumping side effect,
 * unlike getChangedInferenceParams. */
export function pickPersistedInferenceParams(
  params: InferenceParams,
): PersistedInferenceParams {
  const picked: PersistedInferenceParams = {};
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const value = params[key];
    if (value !== undefined) {
      setInferenceParam(picked as InferenceParams, key, value);
    }
  }
  return picked;
}

function hasKeys(value: object): boolean {
  return Object.keys(value).length > 0;
}

/** The map after an edit, or null when there is nothing to record (off, no model,
 * or no persisted param moved). Null leaves the map and its hydration version
 * untouched. */
export function getRememberedParamsPatch(
  enabled: boolean,
  paramsByModel: Record<string, PersistedInferenceParams>,
  modelId: string | undefined,
  changedParams: PersistedInferenceParams,
  snapshot: PersistedInferenceParams,
): Record<string, PersistedInferenceParams> | null {
  if (!(enabled && modelId && hasKeys(changedParams))) {
    return null;
  }
  // The whole snapshot, not just what moved: replay overlays the entry onto the
  // outgoing model's params, so a partial entry would leave the gaps filled by
  // whichever model was on screen last.
  return { ...paramsByModel, [modelId]: snapshot };
}

/** The params a model switch should land on. A model with nothing remembered
 * keeps what is on screen rather than snapping to defaults. Returns `current` by
 * identity when nothing was replayed, so callers can tell the two apart.
 *
 * `maxTokensCap` is the context the model just loaded with, when the caller
 * knows it: a budget remembered from a larger context does not fit. */
export function getReplayedParams(
  enabled: boolean,
  paramsByModel: Record<string, PersistedInferenceParams>,
  current: InferenceParams,
  modelId: string,
  checkpointChanged: boolean,
  maxTokensCap?: number,
): InferenceParams {
  if (!(enabled && checkpointChanged)) {
    return current;
  }
  const remembered = paramsByModel[modelId];
  if (!remembered) {
    return current;
  }
  const replayed = { ...current, ...remembered };
  if (maxTokensCap !== undefined && replayed.maxTokens > maxTokensCap) {
    replayed.maxTokens = maxTokensCap;
  }
  return replayed;
}
