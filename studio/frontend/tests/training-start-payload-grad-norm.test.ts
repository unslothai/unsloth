// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The mapper used to send max_grad_norm: 0.0 on every run, which turns global-norm
// clipping off on MLX and leaves the Gradient Norm chart with no samples to plot.
// Omission is the whole fix, and nothing else fails if it is undone: the field is
// optional in the request type, so restoring the literal would typecheck and pass
// every other suite while silently emptying the chart again.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);

const CONFIG: TrainingConfigState = {
  currentStep: 5,
  modelType: "text",
  selectedModel: "unsloth/gemma-3-270m-it",
  projectName: "grad-norm",
  trainingMethod: "lora",
  hfToken: "",
  datasetSource: "huggingface",
  datasetFormat: "auto",
  dataset: "unsloth/test",
  datasetSubset: null,
  datasetSplit: "train",
  datasetEvalSplit: null,
  datasetStreaming: false,
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
  epochs: 1,
  contextLength: 2048,
  learningRate: 2e-4,
  embeddingLearningRate: null,
  optimizerType: "adamw_8bit",
  lrSchedulerType: "linear",
  loraRank: 16,
  loraAlpha: 16,
  loraDropout: 0,
  loraVariant: "lora",
  batchSize: 2,
  gradientAccumulation: 4,
  weightDecay: 0.001,
  warmupSteps: 5,
  maxSteps: 60,
  saveSteps: 100,
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
  logFrequency: 1,
  isCheckingVision: false,
  isVisionModel: false,
  isEmbeddingModel: false,
  isAudioModel: false,
  isLoadingModelDefaults: false,
  modelDefaultsError: null,
  modelDefaultsAppliedFor: null,
  isCheckingDataset: false,
  isDatasetImage: false,
  isDatasetAudio: false,
  trustRemoteCode: false,
  approvedRemoteCodeFingerprint: null,
  finetuneVisionLayers: false,
  finetuneLanguageLayers: true,
  finetuneAttentionModules: true,
  finetuneMLPModules: true,
  targetModules: ["q_proj", "v_proj"],
  maxPositionEmbeddings: null,
  visionImageSize: null,
  s3Config: null,
};

test("the payload leaves max_grad_norm unset so the backend default governs", () => {
  const payload = buildTrainingStartPayload(CONFIG);

  assert.equal(
    Object.hasOwn(payload, "max_grad_norm"),
    false,
    "max_grad_norm must be absent, not null and not 0",
  );
  // An explicit null would serialize and pin the backend to "no global clipping"
  // just as 0.0 did, so check the wire form too.
  assert.equal("max_grad_norm" in JSON.parse(JSON.stringify(payload)), false);
});

test("the sibling clip knobs keep their existing wire contract", () => {
  const payload = buildTrainingStartPayload(CONFIG);

  assert.equal(payload.max_grad_value, null);
  assert.equal(payload.weight_decay, CONFIG.weightDecay);
});
