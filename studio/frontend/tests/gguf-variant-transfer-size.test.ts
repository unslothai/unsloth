// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A partial row used to print the variant total next to its resume button, so continuing a
// sharded download that was most of the way there still read "56 GB" and looked like the whole
// model coming down again.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { ggufVariantFitRank, ggufVariantTransferBytes, ggufVariantTransferLabel } =
  await import("../src/features/hub/lib/gguf-variant-sort.ts");

const GB = 1000 ** 3;

test("a partial is priced by what is left, not by the variant total", () => {
  const variant = {
    size_bytes: 56 * GB,
    download_size_bytes: 56 * GB,
    download_remaining_bytes: 16 * GB,
    partial: true,
  };

  assert.equal(ggufVariantTransferBytes(variant), 16 * GB);
  assert.equal(ggufVariantTransferLabel(variant), "16 GB left");
});

test("a one-file quant still prices the whole thing, and says so", () => {
  // Nothing to keep: the interrupted file restarts, so "left" is the full size.
  const variant = {
    size_bytes: 18 * GB,
    download_size_bytes: 18 * GB,
    download_remaining_bytes: 18 * GB,
    partial: true,
  };

  assert.equal(ggufVariantTransferLabel(variant), "18 GB left");
});

test("a variant that is not partial shows its plain size", () => {
  const variant = {
    size_bytes: 18 * GB,
    download_size_bytes: 18 * GB,
    download_remaining_bytes: null,
    partial: false,
  };

  assert.equal(ggufVariantTransferBytes(variant), 18 * GB);
  assert.equal(ggufVariantTransferLabel(variant), "18 GB");
});

test("an unmeasured partial falls back to the total, never to a smaller guess", () => {
  for (const remaining of [null, undefined, -1]) {
    const variant = {
      size_bytes: 18 * GB,
      download_size_bytes: 18 * GB,
      download_remaining_bytes: remaining,
      partial: true,
    };
    assert.equal(ggufVariantTransferBytes(variant), 18 * GB);
  }
});

test("a partial with nothing left reads as zero rather than the total", () => {
  const variant = {
    size_bytes: 18 * GB,
    download_size_bytes: 18 * GB,
    download_remaining_bytes: 0,
    partial: true,
  };

  assert.equal(ggufVariantTransferBytes(variant), 0);
});

test("the menu is ranked by the footprint the rows are badged by", () => {
  // Ranking on the weights while badging the total put a "partial" row above "fits".
  const resources = { gpuGb: 16, systemRamGb: 32 };
  const main = 12_789_199_648;
  const row = (over: { download_size_bytes?: number }) => ({
    quant: "Q3_K_S",
    filename: "muse-Q3_K_S.gguf",
    size_bytes: main,
    download_size_bytes: main,
    ...over,
  });

  assert.equal(ggufVariantFitRank(row({}), resources), 0);
  assert.equal(
    ggufVariantFitRank(
      row({ download_size_bytes: main + 3_849_173_920 + 1_451_094_176 }),
      resources,
    ),
    2,
  );
});
