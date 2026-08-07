// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { shouldUseVisionDatasetCheck } from "../src/features/training/lib/fresh-dataset-check.ts";

test("fresh start retries an unclassified image dataset with vision detection", () => {
  const failedBackgroundCheck = {
    datasetCheckFailed: true,
    isDatasetImage: null,
    isVisionModel: true,
  };

  assert.equal(shouldUseVisionDatasetCheck(failedBackgroundCheck), false);
  assert.equal(shouldUseVisionDatasetCheck(failedBackgroundCheck, true), true);
});

test("fresh start does not use vision detection for a text-only model", () => {
  assert.equal(
    shouldUseVisionDatasetCheck(
      { isDatasetImage: null, isVisionModel: false },
      true,
    ),
    false,
  );
});
