// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { TrainingStartRequest } from "../src/features/training/types/api.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  createTrainingStartInputIdentity,
  normalizeTrainingStartPayloadForComparison,
} = await import(
  "../src/features/training/lib/training-start-inputs.ts"
);

function startPayload(
  overrides: Partial<TrainingStartRequest> = {},
): TrainingStartRequest {
  return {
    model_name: "org/model",
    model_known_cached: false,
    model_local_path: null,
    model_format: null,
    hf_dataset: "org/dataset",
    dataset_known_cached: false,
    dataset_local_path: null,
    ...overrides,
  } as TrainingStartRequest;
}

test("cache pins remain part of training start identity", () => {
  const cached = normalizeTrainingStartPayloadForComparison(
    startPayload({
      model_known_cached: true,
      model_local_path: "/cache/model",
      dataset_known_cached: true,
      dataset_local_path: "/cache/dataset",
      model_format: "safetensors",
    }),
  );
  const differentModelCopy = normalizeTrainingStartPayloadForComparison(
    startPayload({
      model_known_cached: true,
      model_local_path: "/cache/other-model",
      dataset_known_cached: true,
      dataset_local_path: "/cache/dataset",
      model_format: "safetensors",
    }),
  );
  const differentDatasetCopy = normalizeTrainingStartPayloadForComparison(
    startPayload({
      model_known_cached: true,
      model_local_path: "/cache/model",
      dataset_known_cached: true,
      dataset_local_path: "/cache/other-dataset",
      model_format: "safetensors",
    }),
  );

  assert.equal(cached.model_known_cached, true);
  assert.equal(cached.model_local_path, "/cache/model");
  assert.equal(cached.dataset_known_cached, true);
  assert.equal(cached.dataset_local_path, "/cache/dataset");
  assert.notDeepEqual(differentModelCopy, cached);
  assert.notDeepEqual(differentDatasetCopy, cached);
  assert.notDeepEqual(
    normalizeTrainingStartPayloadForComparison(startPayload()),
    cached,
  );
});

test("untrainable model formats remain part of training start identity", () => {
  const normalized = normalizeTrainingStartPayloadForComparison(
    startPayload({ model_format: "gguf" }),
  );

  assert.equal(normalized.model_format, "gguf");
});

test("manual draft edits remain part of training start identity before blur", () => {
  const config = {
    modelType: "text" as const,
    isVisionModel: false,
    isAudioModel: false,
    manualDatasetOptionsValid: true,
    userEditRevision: 7,
  };
  const identity = createTrainingStartInputIdentity(startPayload(), config);

  assert.notDeepEqual(
    createTrainingStartInputIdentity(startPayload(), {
      ...config,
      userEditRevision: 8,
    }),
    identity,
  );
  assert.notDeepEqual(
    createTrainingStartInputIdentity(startPayload(), {
      ...config,
      manualDatasetOptionsValid: false,
    }),
    identity,
  );
});
