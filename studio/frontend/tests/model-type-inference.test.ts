// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  inferTrainingModelTypeFromFlags,
  resolveTrainingModelType,
} from "../src/features/training/lib/model-type-capabilities.ts";

test("primary model type prefers vision for audio-capable vision models", () => {
  assert.equal(
    inferTrainingModelTypeFromFlags({ isAudio: true, isVision: true }),
    "vision",
  );
  assert.equal(
    inferTrainingModelTypeFromFlags({
      isEmbedding: true,
      isAudio: true,
      isVision: true,
    }),
    "embeddings",
  );
});

test("backend model types stay authoritative outside the dual-modality conflict", () => {
  assert.equal(
    resolveTrainingModelType({
      modelType: "audio",
      isAudio: true,
      isVision: true,
    }),
    "vision",
  );
  assert.equal(
    resolveTrainingModelType({ modelType: "audio", isAudio: true }),
    "audio",
  );
  assert.equal(
    resolveTrainingModelType({ modelType: "embeddings" }),
    "embeddings",
  );
});
