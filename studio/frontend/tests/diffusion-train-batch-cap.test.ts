// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// MiniMax-H3's forward covers ONE packed sequence, so its backend validation REFUSES a batch
// above 1 rather than clamping. The Train panel rendered Batch unrestricted and always sent the
// field's state, so a 2 typed here -- or simply carried over from the family the user was on a
// moment ago -- came back as a rejected Start with nothing on the control to say why. The panel
// builds its payload inline, so the contract is asserted against the source, the same way the
// labeling-grid gate is in dataset-file-selection.test.ts.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
  "utf8",
);

test("the batch cap comes from the family the backend reported", () => {
  assert.match(source, /max_train_batch_size/);
  assert.match(source, /const batchIsFixed = maxBatchSize != null && maxBatchSize <= 1;/);
  // Clamped, not merely compared: the field is hidden rather than reset, so the cap has to be
  // applied on the way out too.
  assert.match(
    source,
    /const effectiveBatchSize = maxBatchSize == null \? batchSize : Math\.min\(batchSize, maxBatchSize\);/,
  );
});

test("the hidden Batch field cannot still be sent", () => {
  assert.match(source, /train_batch_size: effectiveBatchSize,/);
  assert.doesNotMatch(source, /train_batch_size: batchSize,/);
  // And the control itself is gone for a capped family rather than left offering a value the
  // backend will refuse.
  assert.match(source, /\{!batchIsFixed &&\s*\n\s*numberField\("Batch"/);
});

test("an uncapped family is untouched", () => {
  // max_train_batch_size is null for every other family, and the ?? null / == null pair is what
  // keeps those on the field's own value. An older backend reports nothing, which reads the same.
  assert.match(source, /reportedFamily\?\.max_train_batch_size \?\? null/);
});
