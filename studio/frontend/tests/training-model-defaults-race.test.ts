// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MODEL_DEFAULT_STATE_KEYS,
  trainingConfigPatchTouchesModelDefaults,
} from "../src/features/training/lib/model-defaults-edit-policy.ts";

test("dataset selection does not invalidate pending model defaults", () => {
  assert.equal(
    trainingConfigPatchTouchesModelDefaults({
      datasetSource: "huggingface",
      dataset: "org/dataset",
      datasetSubset: null,
      datasetSplit: null,
      datasetEvalSplit: null,
      datasetKnownCached: false,
      datasetLocalPath: null,
      uploadedFile: null,
      isDatasetImage: null,
      isDatasetAudio: false,
      isCheckingDataset: false,
      datasetCheckFailed: false,
    }),
    false,
  );
});

test("every model-default field invalidates a pending defaults patch", () => {
  for (const key of MODEL_DEFAULT_STATE_KEYS) {
    assert.equal(
      trainingConfigPatchTouchesModelDefaults({ [key]: undefined }),
      true,
      key,
    );
  }
});
