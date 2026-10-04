// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { RecipeExecutionRecord } from "../src/features/recipe-studio/execution-types.ts";
import {
  hasCompleteLocalDataset,
  withExecutionDefaults,
} from "../src/features/recipe-studio/executions/execution-helpers.ts";

test("a legacy cached page without a recorded total remains partial", () => {
  const legacy = withExecutionDefaults({
    id: "legacy-run",
    dataset: Array.from({ length: 20 }, (_, index) => ({ index })),
  } as unknown as RecipeExecutionRecord);

  assert.equal(legacy.datasetTotal, null);
  assert.equal(hasCompleteLocalDataset(legacy), false);
});

test("a cached dataset is complete when its recorded total is present", () => {
  const current = withExecutionDefaults({
    id: "current-run",
    dataset: [{ index: 0 }, { index: 1 }],
    datasetTotal: 2,
  } as unknown as RecipeExecutionRecord);

  assert.equal(current.datasetTotal, 2);
  assert.equal(hasCompleteLocalDataset(current), true);
});
