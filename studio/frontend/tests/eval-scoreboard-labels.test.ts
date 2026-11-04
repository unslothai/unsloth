// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { modelLabels } = await import("../src/features/benchmark/lib/model-labels.ts");

test("two models with one folder name stay two models, labelled with enough of their path", () => {
  const labels = modelLabels([
    "/exports/round-1/adapter",
    "/exports/round-2/adapter",
    "unsloth/Qwen3-0.6B-GGUF:Q8_0",
    "me/Qwen3-0.6B-GGUF:Q8_0",
    "org/solo",
  ]);
  assert.equal(labels["/exports/round-1/adapter"], "round-1/adapter");
  assert.equal(labels["/exports/round-2/adapter"], "round-2/adapter");
  assert.equal(labels["unsloth/Qwen3-0.6B-GGUF:Q8_0"], "unsloth/Qwen3-0.6B-GGUF:Q8_0");
  assert.equal(labels["me/Qwen3-0.6B-GGUF:Q8_0"], "me/Qwen3-0.6B-GGUF:Q8_0");
  assert.equal(labels["org/solo"], "solo");
});

test("the scoreboard keys bars, colours and latest runs on the full model id", () => {
  const src = readFileSync(
    new URL("../src/features/benchmark/components/eval-scoreboard.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(src, /modelName\(/);
  assert.match(src, /const model = run\.model;/);
  assert.match(src, /if \(!models\.includes\(r\.model\)\) models\.push\(r\.model\);/);
});
