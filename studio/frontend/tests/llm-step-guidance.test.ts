// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_HYPERPARAMS } from "../src/config/training.ts";
import {
  MAX_SUGGESTED_STEPS,
  MIN_SUGGESTED_STEPS,
  llmStepGuidance,
  trainedRowCount,
} from "../src/features/training/lib/step-guidance.ts";
import { readSrc } from "./helpers/kit.ts";

test("one pass is ceil(rows / (batch x grad accum)) steps", () => {
  // the default 4 x 8 = 32 rows per step
  assert.deepEqual(llmStepGuidance(10_000, 4, 8), {
    rows: 10_000,
    effectiveBatch: 32,
    passSteps: 313,
    suggestedSteps: 313,
    kind: "pass",
  });
  assert.equal(llmStepGuidance(3200, 4, 8)?.suggestedSteps, 100);
  assert.equal(llmStepGuidance(3201, 4, 8)?.suggestedSteps, 101);
  assert.equal(llmStepGuidance(500, 1, 1)?.suggestedSteps, 500);
});

test("a tiny set gets the floor, a huge one the cap, and the pass count stays exact", () => {
  const tiny = llmStepGuidance(100, 4, 8);
  assert.equal(tiny?.kind, "floor");
  assert.equal(tiny?.passSteps, 4);
  assert.equal(tiny?.suggestedSteps, MIN_SUGGESTED_STEPS);
  const huge = llmStepGuidance(1_000_000, 4, 8);
  assert.equal(huge?.kind, "cap");
  assert.equal(huge?.passSteps, 31_250);
  assert.equal(huge?.suggestedSteps, MAX_SUGGESTED_STEPS);
  assert.equal(llmStepGuidance(MIN_SUGGESTED_STEPS * 32, 4, 8)?.kind, "pass");
  assert.equal(llmStepGuidance(MAX_SUGGESTED_STEPS * 32, 4, 8)?.kind, "pass");
});

test("unknown rows give no guidance and bad batch values count as 1", () => {
  assert.equal(llmStepGuidance(null, 4, 8), null);
  assert.equal(llmStepGuidance(undefined, 4, 8), null);
  assert.equal(llmStepGuidance(0, 4, 8), null);
  assert.equal(llmStepGuidance(Number.NaN, 4, 8), null);
  assert.equal(llmStepGuidance(64, 0, Number.NaN)?.effectiveBatch, 1);
});

test("the trained rows follow the inclusive train slice", () => {
  assert.equal(trainedRowCount(1000), 1000);
  assert.equal(trainedRowCount(1000, null, null), 1000);
  assert.equal(trainedRowCount(1000, "100", null), 900);
  assert.equal(trainedRowCount(1000, null, "99"), 100);
  assert.equal(trainedRowCount(1000, "10", "19"), 10);
  assert.equal(trainedRowCount(1000, " ", "5000"), 1000);
  assert.equal(trainedRowCount(1000, "abc", "-1"), 1000);
  assert.equal(trainedRowCount(1000, "2000", null), null);
  assert.equal(trainedRowCount(1000, "50", "10"), null);
  assert.equal(trainedRowCount(null), null);
  assert.equal(trainedRowCount(0), null);
});

test("the stored default is untouched: guidance only", () => {
  assert.equal(DEFAULT_HYPERPARAMS.maxSteps, 60);
  const hook = readSrc(
    "features/training/hooks/use-max-steps-epochs-toggle.ts",
  );
  assert.match(hook, /const DEFAULT_MAX_STEPS = 60;/);
});

test("the row count comes from the dataset check and is never persisted", () => {
  const store = readSrc("features/training/stores/training-config-store.ts");
  assert.match(
    store,
    /datasetTotalRows:\s*typeof res\.total_rows === "number" && res\.total_rows > 0/,
  );
  const persistence = readSrc(
    "features/training/stores/training-config-persistence.ts",
  );
  assert.match(persistence, /"datasetTotalRows",/);
});

test("the Max Steps field shows the hint and applies it only on click", () => {
  const params = readSrc("features/studio/sections/params-section.tsx");
  assert.match(params, /data-testid="llm-step-guidance"/);
  assert.match(params, /llmStepGuidance\(\s*trainedRowCount\(/);
  assert.match(
    params,
    /onClick=\{\(\) =>\s*store\.setMaxSteps\(stepGuidance\.suggestedSteps\)/,
  );
  assert.equal(params.match(/setMaxSteps\(stepGuidance/g)?.length, 1);
});
