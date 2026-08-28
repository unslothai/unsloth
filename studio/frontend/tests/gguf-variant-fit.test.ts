// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The quant list's fit badge.
 *
 * The case this file exists for: a 90 GB quant on 16 GB of VRAM and 64 GB of RAM
 * badged OOM, and ran at about 12 tok/s. mmap is on by default, so the weights are
 * paged from the file and the sum of volatile memory is not a ceiling. OOM is the
 * one verdict a user acts on by not trying, so it has to mean "cannot", not "slow".
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  classifyGgufVariantFit,
  ggufFitIsAutoSelectable,
} from "../src/features/model-picker/components/model-selector/gguf-variant-fit.ts";

const GB = 1024 ** 3;

/** 0.7 of each, the same shares the picker takes. */
function budget(gpuGb: number, ramGb: number) {
  const gpuBudgetGb = gpuGb * 0.7;
  return {
    gpuBudgetGb,
    totalBudgetGb: gpuBudgetGb + ramGb * 0.7,
    budgetKnown: true,
  };
}

test("a quant inside the GPU share fits", () => {
  assert.equal(classifyGgufVariantFit(10 * GB, budget(24, 64)), "fits");
});

test("over the GPU share but inside RAM is tight", () => {
  assert.equal(classifyGgufVariantFit(20 * GB, budget(24, 64)), "tight");
});

test("past GPU plus RAM pages from disk rather than failing", () => {
  // Reported on the Qwen3.8-Flash-Next thread: UD-Q3_K_XL on a 5060 Ti and 64 GB.
  // Budget is 11.2 + 44.8 = 56 GB, the file is 90 GB, and it ran.
  assert.equal(classifyGgufVariantFit(90 * GB, budget(16, 64)), "disk");
});

test("disk is not auto-selected, so nobody is handed a 169 GB quant", () => {
  assert.equal(ggufFitIsAutoSelectable("fits"), true);
  assert.equal(ggufFitIsAutoSelectable("tight"), true);
  assert.equal(ggufFitIsAutoSelectable("disk"), false);
  assert.equal(ggufFitIsAutoSelectable("oom"), false);
});

test("one memory pool has no disk tier: past it is a real refusal", () => {
  // Unified memory (Mac, and Vulkan hosts reporting no separate GPU pool). There is
  // nothing to offload to, so the tier collapses to fit-or-not against RAM.
  const unified = { gpuBudgetGb: 0, totalBudgetGb: 44.8, budgetKnown: true };
  assert.equal(classifyGgufVariantFit(30 * GB, unified), "fits");
  assert.equal(classifyGgufVariantFit(60 * GB, unified), "oom");
});

test("an unmeasured budget stays permissive, a measured zero does not", () => {
  const unmeasured = { gpuBudgetGb: 0, totalBudgetGb: 0, budgetKnown: false };
  const measuredZero = { gpuBudgetGb: 0, totalBudgetGb: 0, budgetKnown: true };
  assert.equal(classifyGgufVariantFit(100 * GB, unmeasured), "fits");
  assert.equal(classifyGgufVariantFit(100 * GB, measuredZero), "oom");
});

test("a zero-byte listing never reads as an overage", () => {
  assert.equal(classifyGgufVariantFit(0, budget(24, 64)), "fits");
});
