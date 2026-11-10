// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveTrainingStep,
  trainingStepForTourTarget,
  trainingSteps,
} from "../src/features/studio/wizard/training-steps.ts";

test("GRPO sets up the prompt and rewards between the dataset and the settings", () => {
  assert.deepEqual(trainingSteps("grpo"), [
    "model",
    "dataset",
    "prompt",
    "rewards",
    "parameters",
  ]);
});

for (const objective of ["sft", "dpo", "orpo"] as const) {
  test(`${objective} skips the GRPO-only steps`, () => {
    assert.deepEqual(trainingSteps(objective), [
      "model",
      "dataset",
      "parameters",
    ]);
    assert.equal(resolveTrainingStep("dataset", objective), "dataset");
    assert.equal(resolveTrainingStep("rewards", objective), "parameters");
    assert.equal(resolveTrainingStep("prompt", objective), "parameters");
  });
}

test("the guided tour opens the step that holds its anchor", () => {
  assert.equal(trainingStepForTourTarget("studio-model-picker"), "model");
  assert.equal(trainingStepForTourTarget("studio-method"), "model");
  assert.equal(trainingStepForTourTarget("studio-dataset"), "dataset");
  assert.equal(trainingStepForTourTarget("studio-params"), "parameters");
  // The run preview column sits beside every step.
  assert.equal(trainingStepForTourTarget("studio-run-preview"), null);
  assert.equal(trainingStepForTourTarget("studio-nav"), null);
});

test("every tour anchor inside the wizard maps to the step that holds it", async () => {
  const { readFile } = await import("node:fs/promises");
  const wizard = await readFile(
    new URL(
      "../src/features/studio/wizard/training-wizard.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  for (const [, target] of wizard.matchAll(/data(?:-t|T)our="([^"]+)"/g)) {
    assert.notEqual(trainingStepForTourTarget(target), null, target);
  }
});
