// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  getTrainingDatasetRepositoryIds,
  resolveActiveDataset,
} from "./dataset-start-progress.ts";

test("snapshots one dataset and keeps the single-dataset position", () => {
  const ids = getTrainingDatasetRepositoryIds({
    training_datasets: [{ hf_dataset: "org/one" }],
  });
  assert.deepEqual(ids, ["org/one"]);
  assert.deepEqual(resolveActiveDataset(ids, null, null, null), {
    repositoryId: "org/one", index: 1, total: 1,
  });
});

test("uses the explicit active repository in a multiple-dataset run", () => {
  const ids = ["org/one", "org/two"];
  assert.deepEqual(resolveActiveDataset(ids, 2, 2, "org/two"), {
    repositoryId: "org/two", index: 2, total: 2,
  });
});

test("cached first dataset advances from backend progress without a cache percentage", () => {
  const ids = ["org/cached", "org/downloading"];
  assert.equal(resolveActiveDataset(ids, 2, 2, null).repositoryId, "org/downloading");
});

test("remounting during the second download reconstructs the same active dataset", () => {
  const snapshot = ["org/one", "org/two"];
  const before = resolveActiveDataset(snapshot, 2, 2, "org/two");
  const after = resolveActiveDataset(snapshot, 2, 2, "org/two");
  assert.deepEqual(after, before);
});

test("falls back to the legacy dataset field", () => {
  assert.deepEqual(
    getTrainingDatasetRepositoryIds({ training_datasets: [], hf_dataset: "org/legacy" }),
    ["org/legacy"],
  );
});
