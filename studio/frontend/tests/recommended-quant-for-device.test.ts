// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The picker used to recommend whatever quant the repo defaults to, whenever it ran at all. On a
// machine with room for far more, that pinned the suggestion to the default's size: a 64 GiB box
// was told to run UD-Q4_K_XL at 18 GiB while UD-Q6_K_XL at 25 GiB fit with the reserve intact.

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

test("a roomy machine is recommended more than the repo default's size", () => {
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
    assert.ok(!ggufFitIsComfortable(fit(size * GB)), `${size} GiB is comfortable`);
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
  assert.equal(recommendedQuantForDevice([], () => "fits"), null);
});
