// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingConfigState } from "../types/config";

export const TRAINING_TRANSIENT_STATE_KEYS = [
  "isCheckingVision",
  "isVisionModel",
  "isEmbeddingModel",
  "isAudioModel",
  "isLoadingModelDefaults",
  "modelDefaultsError",
  "modelDefaultsAppliedFor",
  "modelDefaultsAppliedKey",
  "isCheckingDataset",
  "isDatasetImage",
  "isDatasetAudio",
  "datasetCheckFailed",
  "datasetMetadataStale",
  "maxPositionEmbeddings",
] as const satisfies readonly (keyof TrainingConfigState)[];

type TrainingTransientStateKey = (typeof TRAINING_TRANSIENT_STATE_KEYS)[number];

export type PersistedTrainingConfigKey = Exclude<
  keyof TrainingConfigState,
  TrainingTransientStateKey
>;

export type PersistedTrainingConfigState = Pick<
  TrainingConfigState,
  PersistedTrainingConfigKey
>;

export function createPersistedStateKeys<
  TState extends object,
  const TTransientKeys extends readonly (keyof TState)[],
>(
  state: TState,
  transientKeys: TTransientKeys,
): Exclude<keyof TState, TTransientKeys[number]>[] {
  const transient = new Set<keyof TState>(transientKeys);
  return (Object.keys(state) as (keyof TState)[]).filter(
    (key): key is Exclude<keyof TState, TTransientKeys[number]> =>
      !transient.has(key),
  );
}

export function createTrainingPersistedStateKeys(
  state: TrainingConfigState,
): PersistedTrainingConfigKey[] {
  return createPersistedStateKeys(state, TRAINING_TRANSIENT_STATE_KEYS);
}

export function pickPersistedTrainingConfigState(
  state: Record<string, unknown>,
  keys: readonly PersistedTrainingConfigKey[],
): Partial<PersistedTrainingConfigState> {
  return Object.fromEntries(
    keys
      .filter((key) => Object.hasOwn(state, key))
      .map((key) => [key, state[key]]),
  ) as Partial<PersistedTrainingConfigState>;
}
