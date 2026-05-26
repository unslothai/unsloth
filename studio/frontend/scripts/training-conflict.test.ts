// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  getTrainingSelectionConflict,
  trainingSelectionConflictEqual,
  type TrainingSelectionTarget,
} from "../src/features/models/lib/training-conflict.ts";

const modelTarget: TrainingSelectionTarget = {
  kind: "model",
  id: "Org/New",
  source: "model",
};

test("training conflict helper returns null for unchanged or empty model draft", () => {
  assert.equal(
    getTrainingSelectionConflict(
      {
        selectedModel: "org/new",
        dataset: null,
        datasetSource: "huggingface",
        uploadedFile: null,
      },
      modelTarget,
    ),
    null,
  );
  assert.equal(
    getTrainingSelectionConflict(
      {
        selectedModel: null,
        dataset: null,
        datasetSource: "huggingface",
        uploadedFile: null,
      },
      modelTarget,
    ),
    null,
  );
});

test("training conflict helper snapshots current model draft", () => {
  assert.deepEqual(
    getTrainingSelectionConflict(
      {
        selectedModel: "Org/Current",
        dataset: null,
        datasetSource: "huggingface",
        uploadedFile: null,
      },
      modelTarget,
    ),
    { kind: "model", id: "Org/Current", source: "model" },
  );
});

test("training conflict helper handles dataset source changes", () => {
  assert.deepEqual(
    getTrainingSelectionConflict(
      {
        selectedModel: null,
        dataset: "Org/CurrentDataset",
        datasetSource: "huggingface",
        uploadedFile: null,
      },
      {
        kind: "dataset",
        id: "/uploads/new.jsonl",
        source: "upload",
      },
    ),
    {
      kind: "dataset",
      id: "Org/CurrentDataset",
      source: "huggingface",
    },
  );
});

test("training conflict snapshots compare exact current draft identity", () => {
  const original = {
    kind: "model" as const,
    id: "Org/Current",
    source: "model" as const,
  };
  assert.equal(trainingSelectionConflictEqual(original, { ...original }), true);
  assert.equal(
    trainingSelectionConflictEqual(original, {
      kind: "model",
      id: "Org/Changed",
      source: "model",
    }),
    false,
  );
});
