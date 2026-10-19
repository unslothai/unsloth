// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Hub GGUF card labels the chat picker's recommended quant, only when not on disk (#7646).

import assert from "node:assert/strict";
import test from "node:test";

import { classifyGgufFit } from "../src/lib/gguf-fit.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { recommendedGgufQuantsByGroup } = await import(
  "../src/features/hub/lib/gguf-recommended-quants.ts"
);

const GB = 1024 ** 3;
const variant = (quant: string, gb: number) => ({
  filename: `model-${quant}.gguf`,
  quant,
  size_bytes: gb * GB,
});

const QUANTS = [
  variant("UD-Q2_K_XL", 9.8),
  variant("UD-Q4_K_XL", 18),
  variant("UD-Q6_K_XL", 25),
  variant("UD-Q8_K_XL", 31),
  variant("BF16", 55),
];

const fitOn = (gpuGb: number, systemRamGb: number) => (sizeBytes: number) =>
  classifyGgufFit(sizeBytes, { gpuGb, systemRamGb });

test("the repo default is recommended wherever it loads", () => {
  assert.deepEqual(
    recommendedGgufQuantsByGroup(QUANTS, "UD-Q4_K_XL", fitOn(80, 128)),
    ["UD-Q4_K_XL"],
  );
});

test("a device that cannot hold the default steps down, never up", () => {
  assert.deepEqual(
    recommendedGgufQuantsByGroup(QUANTS, "UD-Q4_K_XL", fitOn(12, 0)),
    ["UD-Q2_K_XL"],
  );
});

test("without a measured budget the repo default is the pick", () => {
  assert.deepEqual(recommendedGgufQuantsByGroup(QUANTS, "UD-Q4_K_XL", null), [
    "UD-Q4_K_XL",
  ]);
  assert.deepEqual(recommendedGgufQuantsByGroup(QUANTS, null, null), []);
});

test("no variants, no recommendation", () => {
  assert.deepEqual(recommendedGgufQuantsByGroup([], "Q4_K_M", null), []);
  assert.deepEqual(
    recommendedGgufQuantsByGroup([], "Q4_K_M", fitOn(24, 64)),
    [],
  );
});

test("the card shows the label on the list row and the trigger, never on a quant on disk", () => {
  const card = readSrc("features/hub/catalog/gguf-download-card.tsx");
  assert.match(
    card,
    /recommended && !item\.downloaded && !item\.partial && \(\s*<DotTag tone="recommended" label="Recommended" \/>/,
  );
  assert.match(card, /recommended=\{recommendedKeys\.has\(item\.key\)\}/);
  assert.match(
    card,
    /!selected\.downloaded &&\s*!selected\.partial &&\s*selectedVariantKey !== null &&\s*recommendedKeys\.has\(selectedVariantKey\)/,
  );
  assert.match(
    card,
    /\{selectedRecommended && \(\s*<DotTag\s+tone="recommended"/,
  );
  assert.match(
    card,
    /mediaRuntime \|\| !showMemoryBar \|\| !variants\s*\?\s*new Set<string>\(\)/,
  );
});
