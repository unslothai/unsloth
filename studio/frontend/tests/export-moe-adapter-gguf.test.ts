// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An adapter with LoRA on MoE expert weights (Inkling 2, gpt-oss) cannot become a GGUF LoRA: llama.cpp's
// converter stops on "Can not map tensor '...experts.base_layer.weight'". The page offers neither adapter-GGUF
// choice for such a run, and does not size its adapter as a ~100 MB file.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  getEstimatedSize,
  MOE_GGUF_ADAPTER_REASON,
} from "../src/features/export/constants.ts";

const page = readFileSync(
  new URL("../src/features/export/export-page.tsx", import.meta.url),
  "utf8",
);

test("an expert adapter has no made-up size", () => {
  assert.equal(getEstimatedSize("lora", [], null), "~100 MB");
  assert.equal(getEstimatedSize("lora", [], null, true), "");
});

test("the reason names what to do instead", () => {
  assert.match(MOE_GGUF_ADAPTER_REASON, /full model as GGUF/);
});

test("both adapter-GGUF choices are disabled for an expert adapter", () => {
  assert.match(page, /moe_target_parameters === true/);
  const disabledUses = page.match(/moeAdapterBlocksGguf/g) ?? [];
  // Declaration, the reset effect (two conditions and its dependency), two buttons' disabled and title.
  assert.ok(disabledUses.length >= 7, `found ${disabledUses.length}`);
});

test("a GGUF LoRA export does not list the full model's quantizations", () => {
  assert.match(page, /quantLevels=\{ggufAsLora \? \[\] : quantLevels\}/);
  assert.match(page, /quantLevels: ggufAsLora \? \[\] : quantLevels/);
});
