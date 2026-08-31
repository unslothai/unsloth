// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { validateTrainingConfig } = await import(
  "../src/features/training/lib/validation.ts"
);

const validConfig = {
  selectedModel: "org/model",
  modelKnownCached: false,
  modelLocalPath: null,
  modelFormat: null,
  learningRate: 0.0002,
  embeddingLearningRate: null,
  datasetSource: "huggingface" as const,
  dataset: "org/dataset",
  datasetSplit: "train",
  manualDatasetOptionsValid: true,
  uploadedFile: null,
  s3Config: null,
  modelType: "text" as const,
  isVisionModel: false,
  isEmbeddingModel: false,
  isAudioModel: false,
  isDatasetAudio: false,
  loraVariant: "rslora" as const,
  trainingMethod: "qlora" as const,
} as TrainingConfigState;

test("training validation rejects non-positive learning rates", () => {
  assert.deepEqual(
    validateTrainingConfig({ ...validConfig, learningRate: 0 }),
    {
      ok: false,
      errorKey: "studio.training.validation.learningRatePositive",
    },
  );
  assert.deepEqual(
    validateTrainingConfig({ ...validConfig, learningRate: Number.NaN }),
    {
      ok: false,
      errorKey: "studio.training.validation.learningRatePositive",
    },
  );
});

test("training validation accepts a positive learning rate", () => {
  assert.deepEqual(
    validateTrainingConfig({ ...validConfig, learningRate: 0.0002 }),
    { ok: true, errorKey: null },
  );
});

test("training validation requires an explicit split for local cached datasets", () => {
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      datasetKnownCached: true,
      datasetStreaming: false,
      datasetSplit: null,
    }),
    {
      ok: false,
      errorKey: "studio.training.validation.hfDatasetSplitRequired",
    },
  );
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      datasetKnownCached: true,
      datasetStreaming: false,
      datasetSplit: "validation",
    }),
    { ok: true, errorKey: null },
  );
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      datasetKnownCached: false,
      datasetSplit: null,
    }),
    { ok: true, errorKey: null },
  );
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      datasetKnownCached: true,
      datasetStreaming: true,
      datasetSplit: null,
    }),
    { ok: true, errorKey: null },
  );
});

test("training validation blocks an invalid uncommitted manual dataset option", () => {
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      manualDatasetOptionsValid: false,
    }),
    {
      ok: false,
      errorKey: "studio.dataset.selectors.manualInvalid",
    },
  );
});

test("training validation rejects committed split instructions in streaming mode", () => {
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      datasetStreaming: true,
      datasetSplit: "train + validation",
    }),
    {
      ok: false,
      errorKey: "studio.dataset.selectors.manualInvalid",
    },
  );
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      datasetStreaming: true,
      datasetSplit: "train",
    }),
    { ok: true, errorKey: null },
  );
});

test("training validation enforces the CPT embedding learning-rate range", () => {
  for (const embeddingLearningRate of [0, 1, -0.0001, Number.NaN]) {
    assert.deepEqual(
      validateTrainingConfig({
        ...validConfig,
        trainingMethod: "cpt",
        embeddingLearningRate,
      }),
      {
        ok: false,
        errorKey: "studio.training.validation.embeddingLearningRateRange",
      },
    );
  }
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      trainingMethod: "cpt",
      embeddingLearningRate: 0.00002,
    }),
    { ok: true, errorKey: null },
  );
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      trainingMethod: "qlora",
      embeddingLearningRate: 0,
    }),
    { ok: true, errorKey: null },
  );
});

test("training validation keeps local dataset paths out of Hub ID validation", () => {
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      datasetSource: "upload",
      dataset: null,
      uploadedFile: "/datasets/team data/train.jsonl",
    }),
    { ok: true, errorKey: null },
  );
});

test("training validation rejects Hub IDs that backend preflight rejects", () => {
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      selectedModel: "org/team/model",
    }),
    {
      ok: false,
      errorKey: "studio.modelPicker.reasonInvalidHubId",
    },
  );
  assert.deepEqual(
    validateTrainingConfig({
      ...validConfig,
      dataset: "owner/dataset--v2",
    }),
    {
      ok: false,
      errorKey: "studio.datasetPicker.reasonInvalidHubId",
    },
  );
});

test("training validation rejects MLX-incompatible training modes", () => {
  assert.deepEqual(
    validateTrainingConfig({ ...validConfig, trainingMethod: "cpt" }, "mac"),
    {
      ok: false,
      errorKey: "studio.params.notSupportedAppleSilicon",
    },
  );
  assert.deepEqual(
    validateTrainingConfig(
      { ...validConfig, modelType: "embeddings", isEmbeddingModel: true },
      "mac",
    ),
    {
      ok: false,
      errorKey: "studio.params.notSupportedAppleSilicon",
    },
  );
});

test("training validation rejects audio training on MLX", () => {
  assert.deepEqual(
    validateTrainingConfig(
      {
        ...validConfig,
        modelType: "audio",
        isAudioModel: true,
        isDatasetAudio: true,
      },
      "mac",
    ),
    {
      ok: false,
      errorKey: "studio.params.notSupportedAppleSilicon",
    },
  );
  assert.deepEqual(
    validateTrainingConfig({ ...validConfig, isDatasetAudio: true }, "mac"),
    {
      ok: false,
      errorKey: "studio.params.notSupportedAppleSilicon",
    },
  );
});

test("training validation allows audio-capable vision models on MLX with image data", () => {
  assert.deepEqual(
    validateTrainingConfig(
      {
        ...validConfig,
        modelType: "vision",
        isVisionModel: true,
        isAudioModel: true,
      },
      "mac",
    ),
    { ok: true, errorKey: null },
  );
});

test("training validation rejects unsupported LoRA variants on MLX", () => {
  for (const loraVariant of ["loftq", "dora"] as const) {
    assert.deepEqual(
      validateTrainingConfig({ ...validConfig, loraVariant }, "mac"),
      {
        ok: false,
        errorKey: "studio.params.notSupportedAppleSilicon",
      },
    );
    assert.deepEqual(
      validateTrainingConfig({ ...validConfig, loraVariant }, "linux"),
      { ok: true, errorKey: null },
    );
  }
  assert.deepEqual(
    validateTrainingConfig(
      { ...validConfig, trainingMethod: "full", loraVariant: "dora" },
      "mac",
    ),
    { ok: true, errorKey: null },
  );
});

test("training validation keeps CPT and embedding training available off MLX", () => {
  assert.deepEqual(
    validateTrainingConfig({ ...validConfig, trainingMethod: "cpt" }, "linux"),
    { ok: true, errorKey: null },
  );
  assert.deepEqual(
    validateTrainingConfig(
      { ...validConfig, modelType: "embeddings", isEmbeddingModel: true },
      "linux",
    ),
    { ok: true, errorKey: null },
  );
  assert.deepEqual(
    validateTrainingConfig(
      {
        ...validConfig,
        modelType: "audio",
        isAudioModel: true,
        isDatasetAudio: true,
      },
      "linux",
    ),
    { ok: true, errorKey: null },
  );
});
