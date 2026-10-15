// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload } =
  await import("../src/features/training/api/mappers.ts");
const { initialTrainingConfigState } =
  await import("../src/features/training/stores/training-config-policy.ts");

function payloadFor(patch: Partial<TrainingConfigState>) {
  const config: TrainingConfigState = {
    ...initialTrainingConfigState,
    modelType: "text",
    selectedModel: "unsloth/gemma-3-270m-it",
    datasetSource: "huggingface",
    dataset: "unsloth/test",
    ...patch,
  };
  return buildTrainingStartPayload(config, null);
}

test("early stopping is off by default", () => {
  assert.equal(initialTrainingConfigState.earlyStoppingPatience, 0);
  assert.equal(payloadFor({ evalSteps: 0.1 }).early_stopping_patience, 0);
});

test("patience is sent only while evaluation is on", () => {
  assert.equal(
    payloadFor({ evalSteps: 0.1, earlyStoppingPatience: 3 })
      .early_stopping_patience,
    3,
  );
  assert.equal(
    payloadFor({ evalSteps: 0, earlyStoppingPatience: 3 })
      .early_stopping_patience,
    0,
  );
});

test("a fractional, negative or huge field becomes a whole number the backend accepts", () => {
  const sent = (value: number) =>
    payloadFor({ evalSteps: 0.1, earlyStoppingPatience: value })
      .early_stopping_patience;
  assert.equal(sent(2.7), 2);
  assert.equal(sent(-4), 0);
  assert.equal(sent(Number.NaN), 0);
  assert.equal(sent(50000), 1000);
});

test("decision and embedding models never send it", () => {
  assert.equal(
    payloadFor({
      modelType: "decision",
      evalSteps: 0.1,
      earlyStoppingPatience: 3,
    }).early_stopping_patience,
    0,
  );
  assert.equal(
    payloadFor({
      modelType: "embeddings",
      evalSteps: 0.1,
      earlyStoppingPatience: 3,
    }).early_stopping_patience,
    0,
  );
});
