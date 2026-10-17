// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveTrainingStep,
  trainingStepForTourTarget,
  trainingSteps,
} from "../src/features/studio/wizard/training-steps.ts";

test("GRPO configures prompt and rewards before reviewing a run", () => {
  assert.deepEqual(trainingSteps("grpo"), [
    "model", "dataset", "prompt", "rewards", "parameters", "review",
  ]);
});

for (const objective of ["sft", "dpo", "orpo"] as const) {
  test(`${objective} skips GRPO-only setup`, () => {
    const steps = trainingSteps(objective);
    assert.deepEqual(steps, ["model", "dataset", "parameters", "review"]);
    for (const step of steps) {
      assert.equal(resolveTrainingStep(step, "grpo"), step);
      assert.equal(resolveTrainingStep(step, objective), step);
    }
  });

  test(`loading a ${objective} config from a GRPO-only step keeps a visible destination`, () => {
    assert.equal(resolveTrainingStep("rewards", objective), "parameters");
    assert.equal(resolveTrainingStep("prompt", objective), "parameters");
  });
}

test("the guided tour reveals the form panel that contains its target", () => {
  for (const target of ["studio-nav", "studio-base-model", "studio-method"]) {
    assert.equal(trainingStepForTourTarget(target), "model");
  }
  assert.equal(trainingStepForTourTarget("studio-dataset"), "dataset");
  assert.equal(trainingStepForTourTarget("studio-params"), "parameters");
  assert.equal(trainingStepForTourTarget("studio-run-preview"), "review");
});
