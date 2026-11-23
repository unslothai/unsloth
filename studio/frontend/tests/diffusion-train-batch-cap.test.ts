// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// MiniMax-H3 rejects batch > 1 rather than clamping; the panel payload is inline, so test source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const source = await readSrcAsync("features/images/train/diffusion-train-panel.tsx");

test("the batch cap comes from the family the backend reported", () => {
  assert.match(source, /max_train_batch_size/);
  assert.match(source, /const batchIsFixed = maxBatchSize != null && maxBatchSize <= 1;/);
  assert.match(
    source,
    /const effectiveBatchSize = maxBatchSize == null \? batchSize : Math\.min\(batchSize, maxBatchSize\);/,
  );
});

test("the hidden Batch field cannot still be sent", () => {
  assert.match(source, /train_batch_size: effectiveBatchSize,/);
  assert.doesNotMatch(source, /train_batch_size: batchSize,/);
  assert.match(source, /\{!batchIsFixed &&\s*\n\s*numberField\("Batch"/);
});

test("an uncapped family is untouched", () => {
  assert.match(source, /reportedFamily\?\.max_train_batch_size \?\? null/);
});
