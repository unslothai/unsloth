// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { isMissingLocalDatasetCacheError } from "../src/features/training/lib/local-cache-errors.ts";

test("recognizes a missing dataset cache by stable error code", () => {
  assert.equal(
    isMissingLocalDatasetCacheError({ errorCode: "dataset_local_cache_miss" }),
    true,
  );
  assert.equal(
    isMissingLocalDatasetCacheError({ code: "dataset_local_cache_miss" }),
    true,
  );
});

test("does not depend on backend error wording", () => {
  assert.equal(
    isMissingLocalDatasetCacheError(
      new Error("Dataset is not available in the local cache."),
    ),
    false,
  );
  assert.equal(
    isMissingLocalDatasetCacheError({ errorCode: "different_error" }),
    false,
  );
});
