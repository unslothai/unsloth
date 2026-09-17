// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { datasetDisplayName } from "../src/components/resource-picker/dataset-display-name.ts";

test("removes case-insensitive upload hash prefixes", () => {
  assert.equal(
    datasetDisplayName(
      "/uploads/0123456789ABCDEF0123456789ABCDEF_training.jsonl",
    ),
    "training.jsonl",
  );
});

test("uses the dataset directory for generated parquet shards", () => {
  assert.equal(
    datasetDisplayName(
      "C:\\recipes\\support-data\\parquet-files\\part-0.parquet",
    ),
    "support-data",
  );
});

test("keeps ordinary basenames unchanged", () => {
  assert.equal(datasetDisplayName("./datasets/alpaca.json"), "alpaca.json");
  assert.equal(datasetDisplayName("/datasets/alpaca/"), "alpaca");
  assert.equal(
    datasetDisplayName("/uploads/0123456789abcdef0123456789abcdef_"),
    "0123456789abcdef0123456789abcdef_",
  );
});
