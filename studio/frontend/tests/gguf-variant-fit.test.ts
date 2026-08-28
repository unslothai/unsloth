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
  ggufFitIsRefusal,
} from "../src/features/model-picker/components/model-selector/gguf-variant-fit.ts";
import { classifyGgufFit as hubFit } from "../src/lib/gguf-fit.ts";

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

test("one memory pool still pages from disk rather than refusing", () => {
  // Unified memory (Mac, and Vulkan hosts reporting no separate GPU pool). There is
  // nothing to OFFLOAD to, which is not the same as nothing to page from: the file
  // is still on the disk and mmap still works.
  const unified = { gpuBudgetGb: 0, totalBudgetGb: 44.8, budgetKnown: true };
  assert.equal(classifyGgufVariantFit(30 * GB, unified), "fits");
  assert.equal(classifyGgufVariantFit(60 * GB, unified), "disk");
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

// ── the floor under every other verdict ──────────────────────────────────────

test("past free disk is a refusal, because there is nothing to page from", () => {
  // Measured on the A100 box this branch was tested on: 124.7 GB free of 253.1.
  // Every quant from UD-Q5_K_XL up is larger than the space it would land in, so
  // FROM DISK there is a promise about a file that cannot be written.
  const box = { ...budget(80, 134.5), diskFreeGb: 124.7 };
  assert.equal(classifyGgufVariantFit(103 * GB, box), "tight");
  assert.equal(classifyGgufVariantFit(147 * GB, box), "nospace");
  assert.equal(classifyGgufVariantFit(329 * GB, box), "nospace");
});

test("the disk check outranks memory, since placement assumes a local file", () => {
  // Small enough for VRAM outright, still undownloadable. The order matters: ask
  // memory first and this returns `fits` for a file that never arrives.
  const tinyDisk = { ...budget(80, 134.5), diskFreeGb: 4 };
  assert.equal(classifyGgufVariantFit(10 * GB, tinyDisk), "nospace");
});

test("the floor compares in the backend's decimal GB, not GiB", () => {
  // main.py reports disk free divided by 1e9 where memory uses 1024^3. A 130 GB
  // file is 121.1 GiB: compared in GiB it slips under 124.4 free and cannot land.
  const box = { ...budget(80, 134.5), diskFreeGb: 124.4 };
  assert.equal(classifyGgufVariantFit(130e9, box), "nospace");
  assert.equal(hubFit(130e9, { gpuGb: 80, systemRamGb: 167.1, diskFreeGb: 124.4 }), "nospace");
});

test("an unread disk figure abstains rather than refusing everything", () => {
  // 0 is what the probe reports before it has answered, and a machine with
  // genuinely zero bytes free has worse problems than a badge.
  const unread = { ...budget(80, 134.5), diskFreeGb: 0 };
  assert.equal(classifyGgufVariantFit(103 * GB, unread), "tight");
  assert.equal(classifyGgufVariantFit(329 * GB, unread), "disk");
  // Absent entirely is the same as unread: every existing caller omits it.
  assert.equal(classifyGgufVariantFit(329 * GB, budget(80, 134.5)), "disk");
});

test("both refusals share a pill, and neither is auto-selected", () => {
  assert.equal(ggufFitIsRefusal("nospace"), true);
  assert.equal(ggufFitIsRefusal("oom"), true);
  assert.equal(ggufFitIsRefusal("disk"), false);
  assert.equal(ggufFitIsRefusal("tight"), false);
  assert.equal(ggufFitIsAutoSelectable("nospace"), false);
});

test("the floor judges the bytes a download would transfer, not the file size", () => {
  const box = { ...budget(80, 160), diskFreeGb: 124.7 };
  // A resumable partial needs only its remainder: 40e9 bytes left of a 158 GB
  // quant lands in 124.7 GB free, so refusing it would refuse a resume that fits.
  assert.equal(classifyGgufVariantFit(147 * GB, box, false, 40e9), "tight");
  // A fresh fetch carries companion files (mmproj and friends): a 110 GB
  // checkpoint whose full footprint is 130 GB does not fit 124.7 GB free, and
  // judging the checkpoint alone would promise a download that cannot land.
  assert.equal(classifyGgufVariantFit(102 * GB, box, false, 130e9), "nospace");
  // Same rule on the Hub card (120 GiB needs 139 GiB, inside its offload band).
  const hub = { gpuGb: 80, systemRamGb: 160, diskFreeGb: 124.7 };
  assert.equal(hubFit(120 * GB, { ...hub, downloadBytes: 40e9 }), "partial");
  assert.equal(hubFit(102 * GB, { ...hub, downloadBytes: 130e9 }), "nospace");
});

test("a quant already on the machine is never refused for the space it holds", () => {
  // Free space excludes the file's own bytes, so a downloaded 147 GiB quant on a
  // nearly-full disk would read as undownloadable while sitting there loadable.
  const box = { ...budget(80, 134.5), diskFreeGb: 124.7 };
  assert.equal(classifyGgufVariantFit(147 * GB, box, true), "tight");
  // Past memory it still pages, from the copy it already has.
  assert.equal(classifyGgufVariantFit(200 * GB, box, true), "disk");
});

// ── the Hub download card, same bug in different words ───────────────────────

test("the Hub card agrees with the quant list on the same file", () => {
  // Two copies of the rule with different formulas: this one is size * 1.15 + 1 GB
  // against 0.97 of the card and half of RAM. They disagreed on wording ("Won't
  // fit" against "OOM") and now they agree on the verdict, which is the half that
  // matters when both are on screen for one download.
  const budget = { gpuGb: 80, systemRamGb: 167.1 };
  // 68 * 1.15 + 1 = 79.2, over the 0.97 budget of 77.6 but under the raw 80 GiB
  // card, which is what `marginal` means. Unchanged by this diff, pinned so the
  // tier below it cannot be widened by accident.
  assert.equal(hubFit(68 * GB, budget), "marginal");
  assert.equal(hubFit(120 * GB, budget), "partial");
  assert.equal(hubFit(330 * GB, budget), "disk");
});

test("the Hub card pages from disk on a machine with no GPU", () => {
  assert.equal(hubFit(30 * GB, { gpuGb: 0, systemRamGb: 128 }), "ram");
  assert.equal(hubFit(300 * GB, { gpuGb: 0, systemRamGb: 128 }), "disk");
});

test("no budget at all is still a refusal, not a disk load", () => {
  // Nothing was measured, so there is no claim to make about paging either.
  assert.equal(hubFit(30 * GB, { gpuGb: 0, systemRamGb: 0 }), "oom");
});

test("the Hub card carries the same disk floor, with the same escape hatches", () => {
  const budget = { gpuGb: 80, systemRamGb: 167.1, diskFreeGb: 124.7 };
  // Past free disk: undownloadable, whatever memory says.
  assert.equal(hubFit(330 * GB, budget), "nospace");
  // Under the floor (110 GiB is 118.1 decimal GB) the memory ladder is untouched.
  assert.equal(hubFit(110 * GB, budget), "partial");
  // Already on the machine: the floor does not apply.
  assert.equal(hubFit(330 * GB, { ...budget, onDisk: true }), "disk");
  // Unread disk abstains, matching the quant list.
  assert.equal(hubFit(330 * GB, { gpuGb: 80, systemRamGb: 167.1 }), "disk");
});
