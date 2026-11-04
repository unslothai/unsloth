// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  pickDefaultQuant,
  recommendedQuantForDevice,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { hfModelFitsDevice } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";
import { parseMetaTokens } from "../src/features/model-picker/components/model-selector/row-meta.ts";
import {
  audioCppRowFit,
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

test("the fits-on-device filter keeps a row when any of its quants fits the card", () => {
  // The YuE2 catalog row: Q8_0 by default (9.51 GiB), Q4_0 the smallest (7.73 GiB).
  const row = {
    id: "audio-cpp/YuE2-GGUF",
    isGguf: true,
    estimatedSizeBytes: yue2[1].size_bytes,
    estimatedResidentGb: yue2[1].estimated_resident_gb,
    smallestResidentGb: yue2[0].estimated_resident_gb,
  };
  const card = (gpuGb: number) => ({
    memoryTotalGb: gpuGb,
    systemRamAvailableGb: 32,
  });
  const rule = (gpuGb: number) => (bytes: number) =>
    classifyGgufFit(bytes, { gpuGb, systemRamGb: 32 });
  for (const [gpuGb, listed, quant] of [
    // Nothing fits: RAM must not lift the row onto a card every quant reads as oom.
    [6, false, undefined],
    // Only Q4_0 fits: the row stays, and opening it picks Q4_0.
    [8, true, "Q4_0"],
    [10, true, "Q8_0"],
  ] as const) {
    assert.equal(hfModelFitsDevice(row, card(gpuGb)), listed, `${gpuGb} GiB`);
    assert.equal(
      audioCppRowFit(row, gpuGb),
      listed ? "fits" : "oom",
      `${gpuGb} GiB badge`,
    );
    const fitting = yue2.filter(
      (v) => classifyGgufVariantFit(v, { gpuGb, systemRamGb: 32 }) === "fits",
    );
    assert.equal(fitting.length > 0, listed, `${gpuGb} GiB quants`);
    if (quant)
      assert.equal(
        recommendedQuantForDevice(yue2, rule(gpuGb), null, gpuGb)?.quant,
        quant,
      );
  }
  // An older backend sends no smallest estimate: the default variant's decides.
  const { smallestResidentGb: _, ...defaultOnly } = row;
  assert.equal(hfModelFitsDevice(defaultOnly, card(8)), false);
  // Without any estimate the size rule, which counts RAM, still decides.
  const { estimatedResidentGb: __, ...plain } = defaultOnly;
  assert.equal(hfModelFitsDevice(plain, card(6)), true);
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
