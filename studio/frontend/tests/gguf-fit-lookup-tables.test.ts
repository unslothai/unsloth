// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Streamed lookup tables must not be charged as resident. Sizes are real GGUF header figures.

import assert from "node:assert/strict";
import test from "node:test";

import {
  ggufFitRuns,
  pickDefaultQuant,
  recommendedQuantForDevice,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import {
  classifyGgufFit,
  classifyGgufFitWithLookupTables,
  classifyGgufVariantFit,
  ggufVariantLookupTableBytes,
} from "../src/lib/gguf-fit.ts";

const GIB = 1024 ** 3;
// 24 GB card + 64 GB RAM: Flash-Next without the table (54.6 GiB) just clears 23.28 + 32.
const HOST_24_64 = { gpuGb: 24, systemRamGb: 64 };

const flashNext = {
  quant: "UD-Q2_K_XL",
  size_bytes: 73.45 * GIB,
  lookup_table_bytes: 26.82 * GIB,
};
const dsv41 = {
  quant: "UD-IQ1_S",
  size_bytes: 216.98 * GIB,
  lookup_table_bytes: 103.0 * GIB,
};
const gemma3n = { quant: "Q4_K_M", size_bytes: 4.23 * GIB };

test("Flash-Next on 24 GB + 64 GB runs with its table on disk instead of reading OOM", () => {
  assert.equal(classifyGgufFit(flashNext.size_bytes, HOST_24_64), "oom");
  assert.equal(classifyGgufVariantFit(flashNext, HOST_24_64), "streamed");
  assert.equal(ggufFitRuns("streamed"), true);
});

test("DeepSeek-V4.1 UD-IQ1_S on 24 GB + 64 GB stays OOM without the Engram table", () => {
  assert.equal(classifyGgufVariantFit(dsv41, HOST_24_64), "oom");
});

test("a variant with no table keeps today's verdict", () => {
  const host = { gpuGb: 8, systemRamGb: 16 };
  assert.equal(
    classifyGgufVariantFit(gemma3n, host),
    classifyGgufFit(gemma3n.size_bytes, host),
  );
  assert.equal(classifyGgufVariantFit(gemma3n, host), "fits");
  assert.equal(ggufVariantLookupTableBytes(gemma3n), 0);
});

test("a table that would not change the verdict leaves it alone", () => {
  const big = { gpuGb: 192, systemRamGb: 256 };
  assert.equal(classifyGgufVariantFit(flashNext, big), "fits");
});

test("a streamed quant runs but ranks below one that fits with everything resident", () => {
  const fitOf = (bytes: number) => classifyGgufFit(bytes, HOST_24_64);
  const comfortable = { quant: "Q2", size_bytes: 15 * GIB, lookup_table_bytes: 0 };
  assert.equal(recommendedQuantForDevice([flashNext], fitOf), flashNext);
  assert.equal(recommendedQuantForDevice([flashNext, comfortable], fitOf), comfortable);
  const smaller = { quant: "UD-IQ1_S", size_bytes: 30 * GIB, lookup_table_bytes: 0 };
  assert.notEqual(classifyGgufVariantFit(smaller, HOST_24_64), "streamed");
});

test("a resident fit the table only improves keeps its own verdict", () => {
  const fitOf = (bytes: number) =>
    bytes > 60 * GIB ? "partial" : bytes > 40 * GIB ? "marginal" : "fits";
  const variant = { quant: "Q4", size_bytes: 70 * GIB, lookup_table_bytes: 20 * GIB };
  assert.equal(classifyGgufFitWithLookupTables(fitOf, variant), "partial");
});

test("a malformed table size is ignored", () => {
  const bytes = (lookup_table_bytes: number) =>
    ggufVariantLookupTableBytes({ size_bytes: GIB, lookup_table_bytes });
  assert.equal(bytes(Number.NaN), 0);
  assert.equal(bytes(5 * GIB), GIB);
});

test("a bare repo click picks the streamed quant rather than the smallest fallback", () => {
  const variants = [
    { ...flashNext, filename: "Q2/a-00001-of-00003.gguf", downloaded: true },
    { quant: "Q8_0", filename: "Q8/a.gguf", size_bytes: 160 * GIB, lookup_table_bytes: 26.82 * GIB },
  ];
  assert.equal(pickDefaultQuant(variants, "Q8_0", HOST_24_64)?.quant, "UD-Q2_K_XL");
});
