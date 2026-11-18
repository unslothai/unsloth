// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { GgufFitClass } from "../src/lib/gguf-fit.ts";
import { pickerOrderedGgufVariants } from "../src/features/model-picker/components/model-selector/variant-order.ts";

const GB = 1024 ** 3;
const quant = (name: string, gb: number, downloaded = false) => ({
  quant: name,
  filename: `m-${name}.gguf`,
  size_bytes: gb * GB,
  downloaded,
});
// 8 GB fits, up to 12 GB is tight, above that out of memory.
const fitOf = (bytes: number): GgufFitClass =>
  bytes <= 8 * GB ? "fits" : bytes <= 12 * GB ? "ram" : "oom";
const budget = { fitOf, anyBudget: true, budgetKnown: true, cardGb: 0 };

test("both pickers order quants: downloaded, fitting largest first, tight, then out of memory", () => {
  const { ordered, recommended } = pickerOrderedGgufVariants(
    [
      quant("BF16", 16),
      quant("Q8_0", 10),
      quant("Q6_K", 7),
      quant("Q4_K_M", 5),
      quant("Q2_K", 3, true),
    ],
    "Q4_K_M",
    budget,
  );
  assert.deepEqual(
    ordered.map((v) => v.quant),
    ["Q2_K", "Q4_K_M", "Q6_K", "Q8_0", "BF16"],
  );
  // The default fits, so it stays the recommendation and leads its tier.
  assert.deepEqual([...recommended], ["Q4_K_M"]);
});

test("an unknown budget keeps the repo's default as the recommendation", () => {
  const { recommended } = pickerOrderedGgufVariants(
    [quant("Q8_0", 10), quant("Q4_K_M", 5)],
    "Q8_0",
    { ...budget, anyBudget: false, budgetKnown: false },
  );
  assert.deepEqual([...recommended], ["Q8_0"]);
});
