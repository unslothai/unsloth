// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { validateTrainingConfig } from "../src/features/training/lib/validation.ts";
import type { TrainingConfigState } from "../src/features/training/types/config.ts";

const baseConfig: TrainingConfigState = {
  currentStep: 1,
  modelType: "text",
  selectedModel: "unsloth/test-model",
  trainingMethod: "qlora",
  datasetSource: "huggingface",
  datasetFormat: "auto",
  dataset: "org/dataset",
  datasetSubset: null,
  datasetSplit: "train",
  datasetEvalSplit: null,
  datasetManualMapping: {},
  datasetSystemPrompt: "",
  datasetUserTemplate: "",
  datasetAssistantTemplate: "",
  datasetLabelMapping: {},
  datasetAdvisorNotification: null,
  datasetSliceStart: null,
  datasetSliceEnd: null,
  uploadedFile: null,
  uploadedEvalFile: null,
  epochs: 3,
  contextLength: 2048,
  learningRate: 0.0002,
  embeddingLearningRate: null,
  optimizerType: "adamw_8bit",
  lrSchedulerType: "linear",
  loraRank: 16,
  loraAlpha: 32,
  loraDropout: 0.05,
  loraVariant: "lora",
  batchSize: 4,
  gradientAccumulation: 8,
  weightDecay: 0.001,
  warmupSteps: 5,
  maxSteps: 60,
  saveSteps: 0,
  evalSteps: 0,
  packing: false,
  trainOnCompletions: true,
  gradientCheckpointing: "unsloth",
  randomSeed: 3407,
  enableWandb: false,
  wandbToken: "",
  wandbProject: "",
  enableTensorboard: false,
  tensorboardDir: "",
  logFrequency: 10,
  isCheckingVision: false,
  isVisionModel: false,
  isEmbeddingModel: false,
  isAudioModel: false,
  isLoadingModelDefaults: false,
  modelDefaultsError: null,
  modelDefaultsAppliedFor: null,
  modelDefaultsAppliedKey: null,
  isCheckingDataset: false,
  isDatasetImage: false,
  isDatasetAudio: false,
  datasetCheckFailed: false,
  datasetMetadataStale: false,
  modelKnownCached: false,
  modelLocalPath: null,
  modelFormat: null,
  datasetKnownCached: false,
  datasetLocalPath: null,
  trustRemoteCode: false,
  finetuneVisionLayers: false,
  finetuneLanguageLayers: true,
  finetuneAttentionModules: true,
  finetuneMLPModules: true,
  targetModules: [],
  maxPositionEmbeddings: null,
  trainOnCompletionsManuallySet: false,
  learningRateManuallySet: false,
  trainingMethodManuallySet: false,
  yamlLearningRate: undefined,
  datasetFormatBeforeCpt: null,
  datasetFormatAutoForcedByCpt: false,
};

type NumericTrainingConfigField =
  | "contextLength"
  | "batchSize"
  | "gradientAccumulation"
  | "maxSteps"
  | "epochs"
  | "learningRate"
  | "embeddingLearningRate"
  | "weightDecay"
  | "warmupSteps"
  | "saveSteps"
  | "evalSteps"
  | "randomSeed"
  | "logFrequency"
  | "loraRank"
  | "loraAlpha"
  | "loraDropout";

function configWith(overrides: Partial<TrainingConfigState>): TrainingConfigState {
  return { ...baseConfig, ...overrides };
}

function numericOverride(
  field: NumericTrainingConfigField,
  value: number,
): Partial<TrainingConfigState> {
  return { [field]: value } as Partial<TrainingConfigState>;
}

test("training validation accepts a valid training config", () => {
  assert.deepEqual(validateTrainingConfig(baseConfig), { ok: true, message: null });
});

test("training validation rejects invalid dataset slice values", () => {
  for (const value of ["1.5", "-1", "abc"]) {
    assert.equal(
      validateTrainingConfig(configWith({ datasetSliceStart: value })).ok,
      false,
      `datasetSliceStart should reject ${value}`,
    );
    assert.equal(
      validateTrainingConfig(configWith({ datasetSliceEnd: value })).ok,
      false,
      `datasetSliceEnd should reject ${value}`,
    );
  }

  assert.equal(
    validateTrainingConfig(
      configWith({ datasetSliceStart: "10", datasetSliceEnd: "10" }),
    ).ok,
    false,
  );
  assert.equal(
    validateTrainingConfig(
      configWith({ datasetSliceStart: "11", datasetSliceEnd: "10" }),
    ).ok,
    false,
  );
});

test("training validation rejects context length above the known model maximum", () => {
  const result = validateTrainingConfig(
    configWith({ contextLength: 4097, maxPositionEmbeddings: 4096 }),
  );

  assert.equal(result.ok, false);
  assert.equal(
    result.message,
    "Context length cannot exceed this model's maximum of 4,096.",
  );
});

test("training validation rejects non-finite numeric values", () => {
  const fields: NumericTrainingConfigField[] = [
    "contextLength",
    "batchSize",
    "gradientAccumulation",
    "maxSteps",
    "epochs",
    "learningRate",
    "embeddingLearningRate",
    "weightDecay",
    "warmupSteps",
    "saveSteps",
    "evalSteps",
    "randomSeed",
    "logFrequency",
    "loraRank",
    "loraAlpha",
    "loraDropout",
  ];

  for (const field of fields) {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      assert.equal(
        validateTrainingConfig(configWith(numericOverride(field, value))).ok,
        false,
        `${field} should reject ${String(value)}`,
      );
    }
  }
});
