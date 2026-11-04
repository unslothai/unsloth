// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  pickDefaultQuant,
  recommendedQuantForDevice,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { parseMetaTokens } from "../src/features/model-picker/components/model-selector/row-meta.ts";
import {
  classifyGgufFit,
  classifyGgufVariantFit,
  classifyVariantFitOnCard,
} from "../src/lib/gguf-fit.ts";

test("a row shows the load estimate beside the file size, not in its place", () => {
  const parsed = parseMetaTokens("GGUF · 7.7GB · 20.5 GB to run");
  assert.equal(parsed.size, "7.7GB");
  assert.deepEqual(parsed.texts, ["20.5 GB to run"]);
});

const GB = 1024 ** 3;
// YuE2 and ACE-Step on the Hub, with the backend's load-check estimates (#16627).
const yue2 = [
  {
    quant: "Q4_0",
    filename: "yue2-3b-q4_0.gguf",
    size_bytes: 2933424181,
    estimated_resident_gb: 7.73,
  },
  {
    quant: "Q8_0",
    filename: "yue2-3b-q8_0.gguf",
    size_bytes: 4531978293,
    estimated_resident_gb: 9.51,
  },
  {
    quant: "BF16",
    filename: "yue2-3b-bf16.gguf",
    size_bytes: 7529267253,
    estimated_resident_gb: 12.86,
  },
];
const aceQ8 = { size_bytes: 6185468096, estimated_resident_gb: 12.17 };
const card10 = { gpuGb: 10, systemRamGb: 32 };

test("an audio.cpp variant is judged on the load check's estimate, not its file size", () => {
  // The llama.cpp rule reads a 5.8 GiB file as fitting a 10 GiB card; the check needs 12.17 GiB.
  assert.notEqual(classifyGgufFit(aceQ8.size_bytes, card10), "oom");
  assert.equal(classifyGgufVariantFit(aceQ8, card10), "oom");
  assert.equal(classifyGgufVariantFit(yue2[1], card10), "fits");
  assert.equal(classifyGgufVariantFit(yue2[2], card10), "oom");
});

test("without an estimate, or without a known GPU, the llama.cpp rule still decides", () => {
  const plain = { size_bytes: 4 * GB };
  const rule = (bytes: number) => classifyGgufFit(bytes, card10);
  assert.equal(
    classifyVariantFitOnCard(rule, plain, 10),
    classifyGgufFit(4 * GB, card10),
  );
  assert.equal(
    classifyVariantFitOnCard(rule, aceQ8, 0),
    classifyGgufFit(aceQ8.size_bytes, card10),
  );
});

test("the recommended and default quant follow the estimate on a 10 GiB card", () => {
  const rule = (bytes: number) => classifyGgufFit(bytes, card10);
  assert.equal(
    recommendedQuantForDevice(yue2, rule, yue2[2], 10)?.quant,
    "Q8_0",
  );
  assert.equal(recommendedQuantForDevice(yue2, rule, null, 8)?.quant, "Q4_0");
  assert.equal(pickDefaultQuant(yue2, "BF16", card10)?.quant, "Q8_0");
});
