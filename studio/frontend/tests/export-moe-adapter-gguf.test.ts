// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Expert adapters get a hint, not disabled GGUF adapter choices: llama.cpp's converter decides.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  getEstimatedSize,
  MOE_GGUF_ADAPTER_HINT,
} from "../src/features/export/constants.ts";

const page = readFileSync(
  new URL("../src/features/export/export-page.tsx", import.meta.url),
  "utf8",
);

test("an expert adapter has no made-up size", () => {
  assert.equal(getEstimatedSize("lora", [], null), "~100 MB");
  assert.match(page, /estimateMethod === "lora" && moeExpertAdapter\s*\? ""/);
});

test("the hint names what to do instead", () => {
  assert.match(MOE_GGUF_ADAPTER_HINT, /full-model GGUF/);
});

test("an expert adapter gets the hint under both adapter-GGUF choices, not a disabled button", () => {
  assert.match(page, /moe_target_parameters === true/);
  assert.match(page, /loraAsGguf && moeExpertAdapter &&/);
  assert.match(page, /ggufTarget === "lora" && moeExpertAdapter &&/);
  assert.doesNotMatch(page, /disabled=\{moeExpertAdapter\}/);
  assert.doesNotMatch(page, /\|\| moeExpertAdapter/);
});

test("a GGUF LoRA export does not list the full model's quantizations", () => {
  assert.match(page, /quantLevels=\{ggufAsLora \? \[\] : quantLevels\}/);
  assert.match(page, /quantLevels: ggufAsLora \? \[\] : quantLevels/);
});
