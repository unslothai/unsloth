// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveDiffusionTrainingFacts } from "../src/features/images/train/diffusion-train-family-facts.ts";

const klein = {
  params: "4B",
  qlora_vram_gb: 10,
  gated: false,
  note: "",
  base_specs: {
    "black-forest-labs/FLUX.2-klein-base-9B": {
      params: "9B",
      qlora_vram_gb: 18,
    },
    "unsloth/FLUX.2-klein-base-9B": {
      params: "9B",
      qlora_vram_gb: 18,
    },
  },
};

test("shows checkpoint-specific Klein facts for the 9B vendor and mirror ids", () => {
  for (const repo of [
    "black-forest-labs/FLUX.2-klein-base-9B",
    "UNSLOTH/FLUX.2-KLEIN-BASE-9B",
  ]) {
    assert.deepEqual(resolveDiffusionTrainingFacts(klein, repo), {
      params: "9B",
      qlora_vram_gb: 18,
      gated: false,
      note: "",
    });
  }
});

test("keeps the family facts for Klein 4B and unknown custom bases", () => {
  assert.equal(
    resolveDiffusionTrainingFacts(
      klein,
      "black-forest-labs/FLUX.2-klein-base-4B",
    ).qlora_vram_gb,
    10,
  );
  assert.equal(
    resolveDiffusionTrainingFacts(klein, "/models/custom").params,
    "4B",
  );
});
