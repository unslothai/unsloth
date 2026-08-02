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
  isAudioModel: false,
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
