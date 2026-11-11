// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  evalsExpectedModel,
  followResidentGgufVariant,
} from "../src/features/benchmark/lib/model-labels.ts";

test("a selection mirroring the loaded quant follows a same-repo reload", () => {
  const selected = followResidentGgufVariant("Q4_K_M", "Q4_K_M", "Q8_0");
  assert.equal(selected, "Q8_0");
  assert.equal(
    evalsExpectedModel("org/repo-GGUF", selected),
    "org/repo-GGUF:Q8_0",
  );
  assert.equal(followResidentGgufVariant("q4_k_m", "Q4_K_M", "Q8_0"), "Q8_0");
  // Unload then reload: mirrored through the null gap.
  assert.equal(followResidentGgufVariant("Q4_K_M", "Q4_K_M", null), null);
  assert.equal(followResidentGgufVariant(null, null, "Q8_0"), "Q8_0");
});

test("an explicit pick of another quant keeps its pin", () => {
  assert.equal(followResidentGgufVariant("Q2_K", "Q4_K_M", "Q8_0"), "Q2_K");
  assert.equal(followResidentGgufVariant("Q2_K", "Q4_K_M", null), "Q2_K");
  assert.equal(followResidentGgufVariant("Q2_K", null, "Q2_K"), "Q2_K");
});

test("the evals hook applies the follow rule on same-checkpoint residency changes", () => {
  const src = readFileSync(
    new URL("../src/features/benchmark/use-evals-model.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /prevActive !== activeGgufVariant[\s\S]*followResidentGgufVariant\(selected, prevActive, activeGgufVariant\)/,
  );
});
