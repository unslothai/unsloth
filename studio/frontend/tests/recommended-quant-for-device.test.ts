// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// With a repo default the picker recommends that default (UD-Q4_K_XL, else Q4_K_M, else Q4_K_S)
// wherever it loads, and steps down only when it cannot. Ranking by "largest that fits" starred
// the 13.25 GiB F16 of Qwen-Image-2.1-GGUF on any large card. Without a default it still ranks by
// the largest quant the device holds with room to spare.

import assert from "node:assert/strict";
import test from "node:test";

import { classifyGgufFit } from "../src/lib/gguf-fit.ts";
import {
  ggufFitIsComfortable,
  recommendedQuantForDevice,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

const GB = 1024 ** 3;
const variant = (quant: string, gb: number) => ({ quant, size_bytes: gb * GB });

/** The quants on the Qwen3.8-27B listing, in the order the picker holds them. */
const QUANTS = [
  variant("UD-Q2_K_XL", 9.8),
  variant("UD-IQ1_S", 6.2),
  variant("UD-Q4_K_XL", 18),
  variant("BF16", 55),
  variant("UD-Q8_K_XL", 31),
  variant("UD-Q6_K_XL", 25),
];

const fitOn = (gpuGb: number, systemRamGb: number) => (sizeBytes: number) =>
  classifyGgufFit(sizeBytes, { gpuGb, systemRamGb });

test("a comfortable verdict is one that keeps its reserve", () => {
  assert.ok(ggufFitIsComfortable("fits"));
  assert.ok(ggufFitIsComfortable("ram"));
  // These load, but at the edge, so they are not what to suggest.
  assert.ok(!ggufFitIsComfortable("marginal"));
  assert.ok(!ggufFitIsComfortable("partial"));
  assert.ok(!ggufFitIsComfortable("oom"));
});

test("with no repo default, a roomy machine gets the largest comfortable quant", () => {
  const pick = recommendedQuantForDevice(QUANTS, fitOn(0, 64));
  assert.equal(pick?.quant, "UD-Q6_K_XL");
});

test("the pick never exceeds what the device can hold", () => {
  for (const ramGb of [8, 16, 32, 64, 128]) {
    const fit = fitOn(0, ramGb);
    const pick = recommendedQuantForDevice(QUANTS, fit);
    assert.ok(pick, `no pick at ${ramGb} GiB`);
    const runnable = QUANTS.filter((q) => fit(q.size_bytes) !== "oom");
    if (runnable.length > 0) {
      assert.notEqual(
        fit(pick.size_bytes),
        "oom",
        `${ramGb} GiB was handed a quant it cannot load`,
      );
    }
  }
});

test("more memory never recommends a smaller quant", () => {
  let previous = 0;
  for (const ramGb of [8, 16, 32, 64, 128, 256]) {
    const pick = recommendedQuantForDevice(QUANTS, fitOn(0, ramGb));
    assert.ok(pick);
    assert.ok(
      pick.size_bytes >= previous,
      `${ramGb} GiB went backwards from the tier below it`,
    );
    previous = pick.size_bytes;
  }
});

// A vision repo fetches an mmproj beside the weights, so download_size_bytes runs ahead of
// size_bytes. Scoring the checkpoint alone recommended a quant that OOMs once the projector lands.
test("companion weights count against the budget", () => {
  const vision = [
    { quant: "UD-Q4_K_XL", size_bytes: 18 * GB, download_size_bytes: 22 * GB },
    { quant: "UD-Q6_K_XL", size_bytes: 25 * GB, download_size_bytes: 29 * GB },
  ];
  const fit = fitOn(0, 64);
  // 64 GiB of RAM offloads 32 GiB. The larger quant needs 29.75 GiB on its weights but 34.35 GiB
  // with the projector, so only the smaller one actually loads.
  assert.equal(fit(25 * GB), "ram");
  assert.equal(fit(29 * GB), "oom");
  assert.equal(recommendedQuantForDevice(vision, fit)?.quant, "UD-Q4_K_XL");
});

// A listing with no size metadata reports zero, which prices as the bare context allowance and so
// reads comfortable on any device. Ranked by weights it sorts last, but the comfortable pass would
// still reach it once every real quant was only marginal or partial.
test("a variant of unknown size does not outrank one that runs", () => {
  const listing = [
    { quant: "UD-Q8_K_XL", size_bytes: 22 * GB },
    { quant: "UD-Q6_K_XL", size_bytes: 20.5 * GB },
    { quant: "UD-IQ2_XXS", size_bytes: 0 },
  ];
  // A 24 GiB card with 16 GiB of RAM offloads both of these, neither with room to spare: the
  // comfortable ceiling here is 19.4 GiB. An unpriced zero clears it, so it used to win.
  const fit = fitOn(24, 16);
  assert.equal(fit(0), "fits");
  for (const size of [22, 20.5]) {
    assert.ok(
      !ggufFitIsComfortable(fit(size * GB)),
      `${size} GiB is comfortable`,
    );
    assert.notEqual(fit(size * GB), "oom", `${size} GiB does not run`);
  }
  assert.equal(recommendedQuantForDevice(listing, fit)?.quant, "UD-Q8_K_XL");
});

test("a group with no sizes at all is left to the caller", () => {
  const unsized = [
    { quant: "UD-Q4_K_XL", size_bytes: 0 },
    { quant: "UD-Q6_K_XL", size_bytes: 0 },
  ];
  assert.equal(recommendedQuantForDevice(unsized, fitOn(24, 16)), null);
});

test("a machine too small for anything still gets the smallest", () => {
  const pick = recommendedQuantForDevice(QUANTS, () => "oom");
  assert.equal(pick?.quant, "UD-IQ1_S");
});

test("nothing to choose from is not a choice", () => {
  assert.equal(
    recommendedQuantForDevice([], () => "fits"),
    null,
  );
});

// Companions are picked per checkpoint, so download size does not have to rise
// with weights. When nothing runs, ranking the fallback by weights can hand
// back the variant furthest from fitting.
test("when nothing runs, the fallback is the smallest footprint", () => {
  const perVariantCompanions = [
    // Heavier weights, light dependency.
    { quant: "UD-Q6_K_XL", size_bytes: 12 * GB, download_size_bytes: 14 * GB },
    // Lighter weights, but it drags a much larger text encoder along.
    { quant: "UD-Q4_K_XL", size_bytes: 10 * GB, download_size_bytes: 40 * GB },
  ];
  const pick = recommendedQuantForDevice(perVariantCompanions, () => "oom");
  assert.equal(pick?.quant, "UD-Q6_K_XL");
});

test("the footprint fallback keeps the better quant when footprints tie", () => {
  const tied = [
    { quant: "UD-Q6_K_XL", size_bytes: 12 * GB, download_size_bytes: 20 * GB },
    { quant: "UD-Q4_K_XL", size_bytes: 10 * GB, download_size_bytes: 20 * GB },
  ];
  assert.equal(
    recommendedQuantForDevice(tied, () => "oom")?.quant,
    "UD-Q6_K_XL",
  );
});

/** unsloth/Qwen-Image-2.1-GGUF: no UD-Q4_K_XL, so the backend's default is Q4_K_M. */
const QWEN_IMAGE_21 = [
  variant("F16", 13.25),
  variant("Q2_K", 2.3),
  variant("Q3_K_S", 2.54),
  variant("Q3_K_M", 2.95),
  variant("Q3_K_XL", 3.36),
  variant("Q4_K_S", 3.64),
  variant("Q4_K_M", 3.91),
  variant("Q5_K_S", 4.19),
  variant("Q5_K_M", 5.02),
  variant("Q6_K", 5.84),
  variant("Q6_K_XL", 6.26),
  variant("Q8_0", 7.12),
];
const byQuant = (list: typeof QWEN_IMAGE_21, quant: string) =>
  list.find((q) => q.quant === quant) ?? null;

test("a large card is recommended the repo default, not F16", () => {
  const preferred = byQuant(QWEN_IMAGE_21, "Q4_K_M");
  for (const [gpuGb, ramGb] of [
    [24, 64],
    [80, 256],
    [180, 2000],
  ]) {
    const fit = fitOn(gpuGb, ramGb);
    assert.ok(
      ggufFitIsComfortable(fit(13.25 * GB)),
      "F16 fits, so the old rule starred it",
    );
    assert.equal(
      recommendedQuantForDevice(QWEN_IMAGE_21, fit, preferred)?.quant,
      "Q4_K_M",
      `${gpuGb} GiB card`,
    );
  }
});

test("the default stays recommended while it loads at all", () => {
  const preferred = byQuant(QWEN_IMAGE_21, "Q4_K_M");
  // Only runs by spilling to RAM, but it runs.
  const fit = (size: number) => (size <= 3.91 * GB ? "partial" : "oom");
  assert.equal(
    recommendedQuantForDevice(QWEN_IMAGE_21, fit, preferred)?.quant,
    "Q4_K_M",
  );
});

test("a default the device cannot load steps down, never up", () => {
  const preferred = byQuant(QWEN_IMAGE_21, "Q4_K_M");
  // Q4_K_S and smaller fit; the default and everything above it do not.
  const fit = (size: number) => (size <= 3.7 * GB ? "fits" : "oom");
  assert.equal(
    recommendedQuantForDevice(QWEN_IMAGE_21, fit, preferred)?.quant,
    "Q4_K_S",
  );
  // Only a spill fits below the default: the largest of those, not a larger quant that is OOM.
  const tight = (size: number) => (size <= 3.0 * GB ? "partial" : "oom");
  assert.equal(
    recommendedQuantForDevice(QWEN_IMAGE_21, tight, preferred)?.quant,
    "Q3_K_M",
  );
});

test("a UD-Q4_K_XL default is kept on a machine that holds every quant", () => {
  const preferred = byQuant(QUANTS, "UD-Q4_K_XL");
  assert.equal(
    recommendedQuantForDevice(QUANTS, fitOn(0, 256), preferred)?.quant,
    "UD-Q4_K_XL",
  );
});

test("nothing loads with a default either: the smallest footprint", () => {
  const preferred = byQuant(QWEN_IMAGE_21, "Q4_K_M");
  assert.equal(
    recommendedQuantForDevice(QWEN_IMAGE_21, () => "oom", preferred)?.quant,
    "Q2_K",
  );
});

test("a default that is not in this group is ignored", () => {
  const stranger = variant("Q4_K_M", 3.91);
  assert.equal(
    recommendedQuantForDevice(QUANTS, fitOn(0, 64), stranger)?.quant,
    "UD-Q6_K_XL",
  );
});
