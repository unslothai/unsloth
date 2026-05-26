// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingConfigState } from "../types/config";

export type TrainingModelDefaultsPatch = Partial<
  Pick<
    TrainingConfigState,
    | "epochs"
    | "contextLength"
    | "learningRate"
    | "optimizerType"
    | "lrSchedulerType"
    | "loraRank"
    | "loraAlpha"
    | "loraDropout"
    | "loraVariant"
    | "batchSize"
    | "gradientAccumulation"
    | "weightDecay"
    | "warmupSteps"
    | "maxSteps"
    | "saveSteps"
    | "evalSteps"
    | "packing"
    | "trainOnCompletions"
    | "gradientCheckpointing"
    | "randomSeed"
    | "enableWandb"
    | "wandbProject"
    | "enableTensorboard"
    | "tensorboardDir"
    | "logFrequency"
    | "finetuneVisionLayers"
    | "trustRemoteCode"
    | "finetuneLanguageLayers"
    | "finetuneAttentionModules"
    | "finetuneMLPModules"
    | "targetModules"
  >
>;

const TRAINING_DRAFT_MODEL_DEFAULT_KEYS: (keyof TrainingModelDefaultsPatch)[] = [
  "epochs",
  "contextLength",
  "learningRate",
  "optimizerType",
  "lrSchedulerType",
  "loraRank",
  "loraAlpha",
  "loraDropout",
  "loraVariant",
  "batchSize",
  "gradientAccumulation",
  "weightDecay",
  "warmupSteps",
  "maxSteps",
  "saveSteps",
  "evalSteps",
  "packing",
  "trainOnCompletions",
  "gradientCheckpointing",
  "randomSeed",
  "enableWandb",
  "wandbProject",
  "enableTensorboard",
  "tensorboardDir",
  "logFrequency",
  "finetuneVisionLayers",
  "finetuneLanguageLayers",
  "finetuneAttentionModules",
  "finetuneMLPModules",
  "targetModules",
];

export function preserveTrainingDraftFromModelDefaults<
  T extends TrainingModelDefaultsPatch,
>(patch: T): T {
  const next = { ...patch };
  for (const key of TRAINING_DRAFT_MODEL_DEFAULT_KEYS) {
    delete next[key];
  }
  return next;
}
