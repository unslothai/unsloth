// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Hub's dataset filters: modalities (any picked) and a row range over its size categories.

import assert from "node:assert/strict";
import test from "node:test";
import {
  datasetModalities,
  datasetSizeBucket,
  matchesDatasetFilters,
} from "../src/features/hub/lib/dataset-filters.ts";

const tags = ["modality:text", "modality:tabular", "size_categories:10K<n<100K"];
const dataset = { modalities: datasetModalities(tags), sizeBucket: datasetSizeBucket(tags) };

test("reads the Hub's modality and size tags", () => {
  assert.deepEqual(dataset.modalities, ["text", "tabular"]);
  assert.equal(dataset.sizeBucket, 2);
  assert.equal(datasetSizeBucket(["size_categories:n>1T"]), 10);
  assert.equal(datasetSizeBucket(["license:mit"]), undefined);
});

test("any picked modality matches; none picked lists every dataset", () => {
  assert.equal(matchesDatasetFilters(dataset, [], undefined, undefined), true);
  assert.equal(matchesDatasetFilters(dataset, ["image", "text"], undefined, undefined), true);
  assert.equal(matchesDatasetFilters(dataset, ["image"], undefined, undefined), false);
});

test("a size bucket matches when it overlaps the row range", () => {
  // 10K-100K rows.
  assert.equal(matchesDatasetFilters(dataset, [], 1e4, undefined), true);
  assert.equal(matchesDatasetFilters(dataset, [], 1e5, undefined), false);
  assert.equal(matchesDatasetFilters(dataset, [], undefined, 1e4), false);
  assert.equal(matchesDatasetFilters(dataset, [], undefined, 1e6), true);
  // No size published: left out once a range is set.
  assert.equal(matchesDatasetFilters({ modalities: [] }, [], 1e3, undefined), false);
});
