// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { parseYamlRlSettings, serializeConfigToYaml } =
  await import("../src/features/training/lib/yaml-config.ts");
const { DEFAULT_HYPERPARAMS, DEFAULT_RL_SETTINGS } =
  await import("../src/config/training.ts");

const base = { ...DEFAULT_HYPERPARAMS, ...DEFAULT_RL_SETTINGS };

test("an SFT config writes no rl section and reads back as SFT", () => {
  const text = serializeConfigToYaml(base as never, false);
  assert.ok(!text.includes("rl:"));
  assert.deepEqual(parseYamlRlSettings(text), { trainingObjective: "sft" });
});

test("GRPO settings round-trip through the YAML file", () => {
  const state = {
    ...base,
    trainingObjective: "grpo",
    rlBeta: 0.04,
    grpoVariant: "gspo",
    grpoNumGenerations: 8,
    grpoMaxCompletionLength: 200,
    grpoTemperature: 0.9,
    grpoEpsilonHigh: 0.28,
    grpoMaskTruncatedCompletions: true,
    grpoEnableThinking: false,
    grpoSystemPrompt: "Answer in <answer> tags.",
    grpoRewards: [{ name: "exact-answer", weight: 2 }],
  };
  const rl = parseYamlRlSettings(serializeConfigToYaml(state as never, false));
  assert.deepEqual(rl, {
    trainingObjective: "grpo",
    rlBeta: 0.04,
    rlMaxPromptLength: null,
    grpoVariant: "gspo",
    grpoNumGenerations: 8,
    grpoMaxCompletionLength: 200,
    grpoTemperature: 0.9,
    grpoEpsilonHigh: 0.28,
    grpoMaskTruncatedCompletions: true,
    grpoEnableThinking: false,
    grpoSystemPrompt: "Answer in <answer> tags.",
    grpoRewards: [{ name: "exact-answer", weight: 2 }],
  });
});

test("DPO writes only the preference settings", () => {
  const text = serializeConfigToYaml(
    { ...base, trainingObjective: "dpo", rlBeta: 0.2 } as never,
    false,
  );
  assert.ok(!text.includes("num_generations"));
  assert.deepEqual(parseYamlRlSettings(text), {
    trainingObjective: "dpo",
    rlBeta: 0.2,
    rlMaxPromptLength: null,
  });
});

test("an unknown objective or wrong-typed fields don't get applied", () => {
  assert.deepEqual(parseYamlRlSettings("rl:\n  objective: ppo\n"), {
    trainingObjective: "sft",
  });
  assert.deepEqual(
    parseYamlRlSettings(
      "rl:\n  objective: grpo\n  variant: nope\n  num_generations: lots\n  rewards: [{weight: 1}, {name: x}]\n",
    ),
    { trainingObjective: "grpo", grpoRewards: [{ name: "x", weight: 1 }] },
  );
});

test("fractional generation counts and token limits are dropped", () => {
  const out = parseYamlRlSettings(
    "rl:\n  objective: grpo\n  num_generations: 2.5\n  max_prompt_length: 100.5\n  max_completion_length: 64.2\n",
  );
  assert.equal(out.grpoNumGenerations, undefined);
  assert.equal(out.rlMaxPromptLength, undefined);
  assert.equal(out.grpoMaxCompletionLength, undefined);
  assert.equal(
    parseYamlRlSettings("rl:\n  objective: grpo\n  num_generations: 6\n")
      .grpoNumGenerations,
    6,
  );
});
