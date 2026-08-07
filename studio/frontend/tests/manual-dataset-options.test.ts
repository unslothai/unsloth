// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  manualDatasetSplitDefault,
  normalizeManualDatasetOption,
  validateManualDatasetSplit,
  validateManualDatasetSubset,
} from "../src/features/training/lib/manual-dataset-options.ts";

test("manual split defaults preserve remote behavior but require local certainty", () => {
  assert.equal(manualDatasetSplitDefault(false), "train");
  assert.equal(manualDatasetSplitDefault(true), "");
});

test("manual dataset options accept backend-compatible config and split names", () => {
  assert.equal(validateManualDatasetSubset("en_US-v2.1"), null);
  assert.equal(validateManualDatasetSubset("config with spaces"), null);
  assert.equal(validateManualDatasetSubset("v1..v2"), null);
  assert.equal(validateManualDatasetSubset(""), null);
  assert.equal(validateManualDatasetSplit("validation", true), null);
  assert.equal(validateManualDatasetSplit("train.clean", true), null);
  assert.equal(validateManualDatasetSplit("train[:10%]", true), null);
  assert.equal(validateManualDatasetSplit("train[1_000:2_000]", true), null);
  assert.equal(
    validateManualDatasetSplit(
      "test[:-5%](pct1_dropremainder) + train[40%:60%](pct1_dropremainder)",
      true,
    ),
    null,
  );
  assert.equal(validateManualDatasetSplit("", false), null);
  assert.equal(normalizeManualDatasetOption("  validation  "), "validation");
});

test("manual dataset options reject missing, traversing, and unsupported values", () => {
  assert.equal(validateManualDatasetSplit("", true), "required");
  assert.equal(validateManualDatasetSubset("../private"), "invalid");
  assert.equal(validateManualDatasetSubset(".."), "invalid");
  assert.equal(validateManualDatasetSubset("config:name"), "invalid");
  assert.equal(validateManualDatasetSubset("config\nname"), "invalid");
  assert.equal(validateManualDatasetSplit("train/value", true), "invalid");
  assert.equal(validateManualDatasetSplit("train evil", true), "invalid");
  assert.equal(validateManualDatasetSplit("train[10%", true), "invalid");
  assert.equal(validateManualDatasetSplit("train[101%:]", true), "invalid");
  assert.equal(
    validateManualDatasetSplit("train[10:20](closest)", true),
    "invalid",
  );
  assert.equal(
    validateManualDatasetSplit(
      "test[:-5%] + train[40%:60%](pct1_dropremainder)",
      true,
    ),
    "invalid",
  );
  assert.equal(validateManualDatasetSplit("x".repeat(129), true), "too_long");
});
