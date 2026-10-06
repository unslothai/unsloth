// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isHuggingFaceDatasetSelected,
  resolveDeletedLocalDatasetSelection,
  shouldClearMissingLocalDatasetSelection,
} from "../src/features/training/lib/dataset-selection.ts";

test("accepts filename-like Hub ids without treating local paths as Hub datasets", () => {
  assert.equal(
    isHuggingFaceDatasetSelected("huggingface", "owner/data.arrow"),
    true,
  );
  assert.equal(
    isHuggingFaceDatasetSelected("huggingface", "owner/data.jsonl"),
    true,
  );
  assert.equal(isHuggingFaceDatasetSelected("upload", "data.arrow"), false);
  assert.equal(isHuggingFaceDatasetSelected("huggingface", "  "), false);
  for (const path of [
    "/datasets/train",
    "./datasets/train",
    "../datasets/train",
    "~/datasets/train",
    String.raw`C:\datasets\train`,
    String.raw`\\server\datasets\train`,
  ]) {
    assert.equal(
      isHuggingFaceDatasetSelected("huggingface", path),
      false,
      path,
    );
  }
});

test("clears missing recipe selections only after local inventory settles", () => {
  const recipeSelection = {
    source: "upload" as const,
    selectedPath: "/datasets/recipes/recipe_support/parquet-files",
    inventorySettled: true,
    inventoryMatchFound: false,
  };
  assert.equal(
    shouldClearMissingLocalDatasetSelection(recipeSelection),
    true,
  );
  assert.equal(
    shouldClearMissingLocalDatasetSelection({
      ...recipeSelection,
      inventorySettled: false,
    }),
    false,
  );
  assert.equal(
    shouldClearMissingLocalDatasetSelection({
      ...recipeSelection,
      inventoryMatchFound: true,
    }),
    false,
  );
  assert.equal(
    shouldClearMissingLocalDatasetSelection({
      ...recipeSelection,
      selectedPath: String.raw`C:\datasets\uploads\train.jsonl`,
    }),
    false,
  );
});

test("clears only the exact local dataset selection reported missing", () => {
  const selectedPath = "/datasets/uploads/train.jsonl";
  assert.equal(
    resolveDeletedLocalDatasetSelection({
      datasetName: selectedPath,
      source: "upload",
      dataset: null,
      uploadedFile: selectedPath,
    }),
    "upload",
  );
  assert.equal(
    resolveDeletedLocalDatasetSelection({
      datasetName: selectedPath,
      source: "upload",
      dataset: null,
      uploadedFile: "/datasets/uploads/replacement.jsonl",
    }),
    null,
    "a stale 404 must not clear a newer selection",
  );
  assert.equal(
    resolveDeletedLocalDatasetSelection({
      datasetName: "owner/dataset",
      source: "huggingface",
      dataset: "owner/dataset",
      uploadedFile: null,
    }),
    null,
    "a Hub 404 must not clear a valid Hub selection",
  );
  assert.equal(
    resolveDeletedLocalDatasetSelection({
      datasetName: String.raw`C:\datasets\train.jsonl`,
      source: "huggingface",
      dataset: String.raw`C:\datasets\train.jsonl`,
      uploadedFile: null,
    }),
    "huggingface",
  );
});
