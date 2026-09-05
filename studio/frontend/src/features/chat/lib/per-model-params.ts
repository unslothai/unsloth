// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-model parameter memory: a per-checkpoint record alongside the global set, so a switch no
// longer hands the next model the previous one's settings. No store or network imports, so
// the rules stay unit-testable.

import type {
  InferenceParams,
  PersistedInferenceParams,
} from "../types/runtime";

export type PersistedInferenceParamKey = keyof PersistedInferenceParams;

/** Params that persist across a reload. `checkpoint` is the key, not a value. */
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
  "seed",
] as const satisfies readonly PersistedInferenceParamKey[];

/** What the memory records. `maxSeqLength` is left out: the context belongs to the load config,
 *  and a second copy would advertise one never loaded. */
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

/** The remembered subset. No version bump, unlike getChangedInferenceParams. */
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

/** Whether an edit moved anything the memory keeps. */
function movedRememberedParam(changed: PersistedInferenceParams): boolean {
  return REMEMBERED_INFERENCE_PARAM_KEYS.some(
    (key) => changed[key] !== undefined,
  );
}

/** Just the keys an edit moved. What goes to the server for a model that already has an entry:
 *  it merges per key, so a whole snapshot would put this browser's copy of every other key
 *  over another tab's. */
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

/** The map after an edit, or null when there is nothing to record (off, no model, no persisted
 *  param moved), which leaves the map and its version untouched. */
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
  // The whole snapshot: replay overlays the entry, so a partial one would leave the gaps filled
  // by whichever model was on screen last.
  return { ...paramsByModel, [modelId]: snapshot };
}

/** The params a switch should land on. A model with nothing remembered keeps what is on screen,
 *  and `current` comes back by identity so callers can tell that apart. `maxTokensCap` is the
 *  context the model just loaded with. */
export function getReplayedParams(
  enabled: boolean,
  paramsByModel: Record<string, PersistedInferenceParams>,
  current: InferenceParams,
  modelId: string,
  checkpointChanged: boolean,
  maxTokensCap?: number,
): InferenceParams {
  // The cap describes the load, not the memory, so it applies even when nothing replays. Identity
  // survives when the budget already fits.
  const capped = (params: InferenceParams): InferenceParams =>
    maxTokensCap !== undefined && params.maxTokens > maxTokensCap
      ? { ...params, maxTokens: maxTokensCap }
      : params;
  if (!(enabled && checkpointChanged)) {
    return capped(current);
  }
  const remembered = paramsByModel[modelId];
  if (!remembered) {
    return capped(current);
  }
  // Key by key, not a spread: the row accepts every persisted key from any writer, and a spread
  // would carry maxSeqLength and stale keys into params.
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
