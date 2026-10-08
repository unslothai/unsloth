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

const BASE: TrainingConfigState = {
  ...initialTrainingConfigState,
  modelType: "text",
  selectedModel: "unsloth/Qwen3-4B-Base",
  trainingMethod: "lora",
  datasetSource: "huggingface",
  dataset: "openai/gsm8k",
  datasetSplit: "train",
  datasetManualMapping: { question: "user", answer: "assistant" },
  rlRoleMapping: { question: "prompt", answer: "answer" },
};

test("SFT payload is unchanged: objective sft, chat mapping, no GRPO rewards", () => {
  const payload = buildTrainingStartPayload(BASE, null);
  assert.equal(payload.objective, "sft");
  assert.deepEqual(payload.custom_format_mapping, {
    question: "user",
    answer: "assistant",
  });
  assert.deepEqual(payload.grpo_rewards, []);
  assert.equal(payload.rl_beta, null);
  assert.equal(payload.rl_system_prompt, null);
});

test("GRPO sends RL roles instead of chat roles, plus its rewards", () => {
  const payload = buildTrainingStartPayload(
    {
      ...BASE,
      trainingObjective: "grpo",
      trainOnCompletions: true,
      rlBeta: 0.04,
      grpoRewards: [{ name: "exact-answer", weight: 2 }],
    },
    null,
  );
  assert.equal(payload.objective, "grpo");
  assert.deepEqual(payload.custom_format_mapping, {
    question: "prompt",
    answer: "answer",
  });
  assert.deepEqual(payload.grpo_rewards, [{ name: "exact-answer", weight: 2 }]);
  assert.equal(payload.rl_beta, 0.04);
  assert.equal(payload.train_on_completions, false);
  assert.equal(payload.grpo_variant, "dapo");
  assert.equal(payload.grpo_enable_thinking, false);
  assert.match(payload.rl_system_prompt ?? "", /<reasoning>/);
});

test("GRPO sends the loss variant and its clip/mask options; DPO drops them", () => {
  const grpo = buildTrainingStartPayload(
    {
      ...BASE,
      trainingObjective: "grpo",
      grpoVariant: "gspo",
      grpoMaskTruncatedCompletions: true,
      grpoEpsilonHigh: 0.28,
    },
    null,
  );
  assert.equal(grpo.grpo_variant, "gspo");
  assert.equal(grpo.grpo_mask_truncated_completions, true);
  assert.equal(grpo.grpo_epsilon_high, 0.28);
  const dpo = buildTrainingStartPayload(
    {
      ...BASE,
      trainingObjective: "dpo",
      grpoMaskTruncatedCompletions: true,
      grpoEpsilonHigh: 0.28,
    },
    null,
  );
  assert.equal(dpo.grpo_mask_truncated_completions, false);
  assert.equal(dpo.grpo_epsilon_high, null);
});

test("DPO does not send GRPO-only fields", () => {
  const payload = buildTrainingStartPayload(
    { ...BASE, trainingObjective: "dpo", grpoMaxCompletionLength: 512 },
    null,
  );
  assert.equal(payload.objective, "dpo");
  assert.deepEqual(payload.grpo_rewards, []);
  assert.equal(payload.grpo_max_completion_length, null);
});

test("CPT always trains as SFT even if a stale objective is persisted", () => {
  const payload = buildTrainingStartPayload(
    { ...BASE, trainingMethod: "cpt", trainingObjective: "grpo" },
    null,
  );
  assert.equal(payload.objective, "sft");
  assert.deepEqual(payload.grpo_rewards, []);
});

test("embedding and audio models always start as SFT", async () => {
  const { effectiveTrainingObjective, rlObjectiveSupported } =
    await import("../src/features/training/lib/rl-roles.ts");
  for (const modelType of ["embeddings", "audio", "decision"]) {
    const state = { modelType, trainingObjective: "grpo" as const };
    assert.equal(rlObjectiveSupported(state), false, modelType);
    assert.equal(effectiveTrainingObjective(state), "sft", modelType);
  }
  assert.equal(
    effectiveTrainingObjective({ modelType: "text", trainingObjective: "dpo" }),
    "dpo",
  );
  assert.equal(
    effectiveTrainingObjective({
      modelType: "text",
      isEmbeddingModel: true,
      trainingObjective: "dpo",
    }),
    "sft",
  );
});
