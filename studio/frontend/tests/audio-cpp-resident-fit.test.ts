// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  pickDefaultQuant,
  recommendedQuantForDevice,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import {
  hfModelFitsDevice,
  loadScopedGpu,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";
import { parseMetaTokens } from "../src/features/model-picker/components/model-selector/row-meta.ts";
import {
  audioCppCardGb,
  audioCppRowFit,
  classifyGgufFit,
  classifyGgufVariantFit,
  classifyVariantFitOnCard,
} from "../src/lib/gguf-fit.ts";
import { readSrc } from "./helpers/kit.ts";

test("a row shows the load estimate beside the file size, not in its place", () => {
  const parsed = parseMetaTokens("GGUF · 7.7GB · 20.5 GB to run");
  assert.equal(parsed.size, "7.7GB");
  assert.equal(parsed.resident, "20.5 GB");
  assert.deepEqual(parsed.texts, []);
});

const GB = 1024 ** 3;
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
  const { smallestResidentGb: _, ...defaultOnly } = row;
  assert.equal(hfModelFitsDevice(defaultOnly, card(8)), false);
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

test("an audio.cpp row is judged on the largest card, where automatic placement loads it", () => {
  // 10 GiB GPU0 beside 24 GiB GPU1: scoping narrows to GPU0, but the load puts ACE-Step on GPU1.
  const host = {
    available: true,
    memoryTotalGb: 34,
    maxDeviceMemoryGb: 24,
    loadDeviceMemoryGb: 10,
    systemRamAvailableGb: 32,
    deviceCount: 2,
  };
  const aceStep = {
    id: "audio-cpp/ACE-Step-1.5-GGUF",
    isGguf: true,
    estimatedResidentGb: 16.36,
    smallestResidentGb: aceQ8.estimated_resident_gb,
  };
  const scoped = loadScopedGpu(host, true);
  assert.equal(scoped.memoryTotalGb, 10);
  const card = audioCppCardGb(host);
  assert.equal(card, 24);
  assert.equal(
    hfModelFitsDevice(aceStep, scoped, { audioCppCardGb: card }),
    true,
  );
  // Two 8 GiB cards: the Hub passes their 16 GiB sum, but no one card holds YuE2 BF16 (12.86 GiB).
  const twoCards = {
    gpuGb: 16,
    gpuCount: 2,
    audioCppCardGb: 8,
    systemRamGb: 32,
  };
  assert.equal(classifyGgufVariantFit(yue2[2], twoCards), "oom");
  assert.equal(classifyGgufVariantFit(yue2[0], twoCards), "fits");
});

test("a CPU RAM load leaves audio.cpp quants to the RAM-aware rule", () => {
  const host = { available: true, maxDeviceMemoryGb: 6 };
  const card = audioCppCardGb(host, true);
  assert.equal(card, 0);
  const budget = { gpuGb: 6, systemRamGb: 64 };
  const rule = (bytes: number) => classifyGgufFit(bytes, budget);
  for (const variant of yue2) {
    assert.equal(
      classifyVariantFitOnCard(rule, variant, audioCppCardGb(host)),
      "oom",
    );
    assert.equal(
      classifyVariantFitOnCard(rule, variant, card),
      classifyGgufFit(variant.size_bytes, budget),
    );
  }
  const plain = yue2.map(({ estimated_resident_gb: _, ...rest }) => rest);
  assert.equal(
    recommendedQuantForDevice(yue2, rule, null, card)?.quant,
    recommendedQuantForDevice(plain, rule, null)?.quant,
  );
  assert.equal(
    classifyGgufVariantFit(yue2[2], { ...budget, audioCppCardGb: card }),
    classifyGgufFit(yue2[2].size_bytes, budget),
  );
  const row = {
    id: "audio-cpp/Yue2-3B-GGUF",
    isGguf: true,
    estimatedSizeBytes: yue2[1].size_bytes,
    estimatedResidentGb: yue2[1].estimated_resident_gb,
    smallestResidentGb: yue2[0].estimated_resident_gb,
  };
  const gpu = { memoryTotalGb: 6, systemRamAvailableGb: 64 };
  assert.equal(hfModelFitsDevice(row, gpu, { audioCppCardGb: 6 }), false);
  assert.equal(hfModelFitsDevice(row, gpu, { audioCppCardGb: card }), true);
});

test("the picker judges audio.cpp variants on the card its load lands on", () => {
  // The quant expander and row badges render only under React, so their wiring is read from source.
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(pickers, /const audioCardGb = audioCppCardGb\(gpu, cpuLoad\);/);
  assert.match(
    readSrc("features/hub/hub-page.tsx"),
    /audioCppCardGb\(gpu, audioDevice === "cpu"\)/,
  );
  assert.match(pickers, /const audioCardGb = useContext\(AudioCppCard\);/);
  assert.match(
    pickers,
    /const residentGb = audioCardGb > 0 \? v\.estimated_resident_gb : null;/,
  );
  assert.doesNotMatch(pickers, /audioCppRowFit\([^)]*memoryTotalGb\)/);
  assert.match(
    readSrc("features/audio/audio-page.tsx"),
    /cpuLoad=\{audioDevice === "cpu"\}/,
  );
});
