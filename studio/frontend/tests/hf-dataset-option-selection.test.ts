// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { nextHfDatasetOptionSelection } from "../src/features/training/lib/hf-dataset-option-selection.ts";

test("metadata selection prefers default config and train when available", () => {
  assert.deepEqual(
    nextHfDatasetOptionSelection({
      subsets: ["english", "default"],
      splits: [],
      selectedSubset: null,
      selectedSplit: null,
    }),
    { type: "subset", value: "default" },
  );
  assert.deepEqual(
    nextHfDatasetOptionSelection({
      subsets: ["english", "default"],
      splits: ["validation", "train"],
      selectedSubset: "default",
      selectedSplit: null,
    }),
    { type: "split", value: "train" },
  );
});

test("metadata selection uses a non-train split when it is the available choice", () => {
  assert.deepEqual(
    nextHfDatasetOptionSelection({
      subsets: ["offline"],
      splits: ["validation"],
      selectedSubset: "offline",
      selectedSplit: null,
    }),
    { type: "split", value: "validation" },
  );
  assert.equal(
    nextHfDatasetOptionSelection({
      subsets: ["offline"],
      splits: ["validation"],
      selectedSubset: "offline",
      selectedSplit: "validation",
    }),
    null,
  );
});

test("metadata selection prefers a train split over a test split listed first", () => {
  assert.deepEqual(
    nextHfDatasetOptionSelection({
      subsets: ["default"],
      splits: ["test_prefs", "train_prefs"],
      selectedSubset: "default",
      selectedSplit: null,
    }),
    { type: "split", value: "train_prefs" },
  );
});
