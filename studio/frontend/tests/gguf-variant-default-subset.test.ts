// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * defaultVariant is a QUANT key picked across the WHOLE repo, and a filenamePrefix subset need
 * not contain it: the H3 catalog offers 6 ref2va quants against 8 fl2va ones. Stored unchanged,
 * effectiveRecommended's budget-unavailable branch returns a key with no visible row, so nothing
 * reads as recommended and the sorter falls through to largest-first.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const source = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("the stored default is dropped when the prefix filters it out", () => {
  const call = source.slice(
    source.indexOf("setDefaultVariant("),
    source.indexOf("setHasVision(normalized.hasVision);"),
  );
  // Conditional on the prefix, and on the default surviving it.
  assert.match(call, /filenamePrefix &&/);
  assert.match(call, /variant\.quant === normalized\.defaultVariant/);
  assert.match(call, /\?\s*null/);
  // Unfiltered listings keep the repo-wide default exactly as before.
  assert.match(call, /:\s*normalized\.defaultVariant/);
});

test("the recommendation still reads the stored default, so clearing it is what takes effect", () => {
  const recommended = source.slice(
    source.indexOf("const effectiveRecommended = useMemo("),
    source.indexOf("const sortedVariants = useMemo("),
  );
  // The branch the bug showed up in: no budget, so the stored key is returned unchecked.
  assert.match(recommended, /\(totalBudgetGb <= 0 && !budgetKnown\)\s*\)\s*\{\s*return defaultVariant;/);
});
