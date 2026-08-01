// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { TrainingStartRequest } from "../src/features/training/types/api.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizeTrainingStartPayloadForComparison } = await import(
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

test("cache reconciliation metadata does not change training start identity", () => {
  const uncached = normalizeTrainingStartPayloadForComparison(startPayload());
  const cached = normalizeTrainingStartPayloadForComparison(
    startPayload({
      model_known_cached: true,
      model_local_path: "/cache/model",
      dataset_known_cached: true,
      dataset_local_path: "/cache/dataset",
      model_format: "safetensors",
    }),
  );

  assert.deepEqual(cached, uncached);
});

test("untrainable model formats remain part of training start identity", () => {
  const normalized = normalizeTrainingStartPayloadForComparison(
    startPayload({ model_format: "gguf" }),
  );

  assert.equal(normalized.model_format, "gguf");
});
