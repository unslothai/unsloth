// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PersistedInferenceParams } from "../types/runtime";

export type PersistedInferenceParamKey = keyof PersistedInferenceParams;

/** Params that persist across a reload. `checkpoint` is the key, not a value. */
export const PERSISTED_INFERENCE_PARAM_KEYS = [
  "samplingFieldsExplicit",
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

/** What per-model memory records. Context length belongs to the load config. */
export const REMEMBERED_INFERENCE_PARAM_KEYS =
  PERSISTED_INFERENCE_PARAM_KEYS.filter(
    (key): key is PersistedInferenceParamKey => key !== "maxSeqLength",
  );
