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
  datasetSource: "huggingface" as const,
  dataset: "org/dataset",
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
