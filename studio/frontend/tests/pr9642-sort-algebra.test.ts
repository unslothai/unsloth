// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// S3 for PR #9642. The same body runs in chromium, firefox and webkit under S8.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { canonicalRecentOrder, runSortAlgebraChecks } = await import(
  "./helpers/pr9642-sort-algebra.ts"
);

test("S3: compareInventoryItemsByRecent is a valid total preorder", () => {
  const failures = runSortAlgebraChecks();
  assert.deepEqual(failures, [], failures.join("\n"));
});

test("S3: the canonical fixture has one unambiguous order", () => {
  // S8 asserts each browser reproduces exactly this list.
  assert.deepEqual(canonicalRecentOrder(), [
    "cache:gguf:Org/already-millis",
    "local:lmstudio-newest",
    "local:lmstudio-mid",
    "cache:gguf:Org/gemma-3",
    "cache:gguf:Org/bert-base-uncased",
    "cache:gguf:Org/no-timestamp",
  ]);
});
