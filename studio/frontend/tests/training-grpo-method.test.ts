// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// GRPO is the fourth training method: it has to map to the backend's "GRPO"
// training_type, carry its reward selection and rollout settings on the wire,
// and stay off the MLX backend, which has no GRPO path.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload, trainingLoadsIn4Bit } = await import(
  "../src/features/training/api/mappers.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const {
  isTrainingMethodSupportedOnDevice,
  parseBackendTrainingMethod,
  toBackendTrainingType,
} = await import("../src/features/training/lib/training-methods.ts");
const { isAdapterMethod, isTrainingMethod } = await import(
  "../src/types/training.ts"
);
const { TRAINING_METHOD_META, TRAINING_METHOD_ORDER } = await import(
  "../src/features/training/lib/training-method-meta.ts"
);

const CONFIG: TrainingConfigState = {
  ...initialTrainingConfigState,
  modelType: "text",
  selectedModel: "unsloth/gemma-3-1b-it",
  trainingMethod: "grpo",
  datasetSource: "huggingface",
  dataset: "openai/gsm8k",
  datasetSplit: "train",
  epochs: 1,
  batchSize: 1,
  gradientAccumulation: 4,
  trainOnCompletions: true,
};

test("grpo is a first-class training method", () => {
  assert.equal(isTrainingMethod("grpo"), true);
  assert.equal(isAdapterMethod("grpo"), true);
  assert.equal(toBackendTrainingType("grpo"), "GRPO");
  assert.equal(parseBackendTrainingMethod("GRPO", false), "grpo");
  assert.ok(TRAINING_METHOD_ORDER.includes("grpo"));
  assert.ok(TRAINING_METHOD_META.grpo);
});

test("grpo is unavailable on the MLX backend", () => {
  assert.equal(isTrainingMethodSupportedOnDevice("grpo", "mac"), false);
  assert.equal(isTrainingMethodSupportedOnDevice("grpo", "cuda"), true);
});

test("the payload carries the reward selection and rollout settings", () => {
  const payload = buildTrainingStartPayload(CONFIG, null);

  assert.equal(payload.training_type, "GRPO");
  assert.equal(payload.use_lora, true);
  assert.deepEqual(payload.reward_functions, CONFIG.rewardFunctions);
  assert.ok((payload.reward_functions ?? []).length > 0);
  assert.equal(payload.num_generations, CONFIG.numGenerations);
  assert.equal(payload.max_prompt_length, CONFIG.maxPromptLength);
  assert.equal(payload.max_completion_length, CONFIG.maxCompletionLength);
  assert.equal(payload.grpo_temperature, CONFIG.grpoTemperature);
  assert.equal(payload.grpo_top_p, CONFIG.grpoTopP);
  assert.equal(payload.grpo_beta, CONFIG.grpoBeta);
  // GRPO scores whole rollouts, so completion masking never applies.
  assert.equal(payload.train_on_completions, false);
});

test("supervised runs never send a reward selection", () => {
  const payload = buildTrainingStartPayload(
    { ...CONFIG, trainingMethod: "lora" },
    null,
  );

  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.deepEqual(payload.reward_functions, []);
});

test("grpo follows the CPT rule for 4-bit loads", () => {
  assert.equal(trainingLoadsIn4Bit(CONFIG), false);
  assert.equal(
    trainingLoadsIn4Bit({
      ...CONFIG,
      selectedModel: "unsloth/gemma-3-1b-it-bnb-4bit",
    }),
    true,
  );
});
