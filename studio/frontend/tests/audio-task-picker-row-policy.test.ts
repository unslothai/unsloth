// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { taskPickerRowMatches } from "../src/features/model-picker/components/model-selector/audio-picker-policy.ts";

test("curated task seeds survive missing Hub metadata and device format gates", () => {
  assert.equal(
    taskPickerRowMatches({
      isCatalogSeed: true,
      format: "all",
      matchesFormat: false,
      matchesTask: false,
      isRecommendable: false,
    }),
    true,
  );
});

test("explicit format choices still filter curated task seeds", () => {
  assert.equal(
    taskPickerRowMatches({
      isCatalogSeed: true,
      format: "gguf",
      matchesFormat: false,
      matchesTask: false,
      isRecommendable: false,
    }),
    false,
  );
});

test("non-curated rows still require task and format compatibility", () => {
  assert.equal(
    taskPickerRowMatches({
      isCatalogSeed: false,
      format: "all",
      matchesFormat: true,
      matchesTask: false,
      isRecommendable: true,
    }),
    false,
  );
  assert.equal(
    taskPickerRowMatches({
      isCatalogSeed: false,
      format: "all",
      matchesFormat: true,
      matchesTask: true,
      isRecommendable: false,
    }),
    false,
  );
});
