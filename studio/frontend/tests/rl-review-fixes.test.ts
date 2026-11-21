// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { formatAxisMetric, formatLoss, rewardLogLabel, smoothSeries } =
  await import("../src/features/studio/sections/charts/utils.ts");
const { rewardTryColumns } = await import(
  "../src/features/flows/editor/reward-roles.ts"
);
const { parseRewardWeight } = await import(
  "../src/features/training/lib/reward-summary.ts"
);
const { parseYamlRlSettings } = await import(
  "../src/features/training/lib/yaml-config.ts"
);
const { isRlRole, resolveRlMapping, streamingBlockersClearChoice } = await import(
  "../src/features/training/lib/rl-roles.ts"
);

test("a GRPO-sized loss reads as a number, not 0.0000, on the stat and the axis", () => {
  assert.equal(formatLoss(0.0000123), "1.23e-5");
  assert.equal(formatLoss(1.38571), "1.3857");
  assert.equal(formatLoss(0), "0.0000");
  const ticks = [0.00001, 0.00002, 0.00003].map(formatAxisMetric);
  assert.deepEqual(ticks, ["1e-5", "2e-5", "3e-5"]);
  assert.equal(new Set(ticks).size, 3);
});

test("the reward legend names match the rewards list", () => {
  assert.equal(rewardLogLabel("rewards/exact_answer/mean"), "exact-answer");
  assert.equal(rewardLogLabel("rewards/soft_xml_format"), "soft-xml-format");
});

test("RL charts get Training Loss's smoothed line over the series", () => {
  const rows = [{ reward: -1 }, {}, { reward: 3 }, { reward: -1 }] as Record<
    string,
    number
  >[];
  const trend = smoothSeries(rows, "reward", 0.6);
  assert.equal(trend[1], undefined);
  assert.equal(trend[0], -1);
  // The EMA sits between the noisy last step and the earlier ones.
  const last = trend[3] ?? Number.NaN;
  assert.ok(last > -1 && last < 3, String(last));
  assert.deepEqual(smoothSeries(rows, "reward", 0), [-1, undefined, 3, -1]);
});

test("an RL objective alone keeps the stored streaming choice for SFT", () => {
  assert.equal(streamingBlockersClearChoice(["rlObjective"]), false);
  assert.equal(streamingBlockersClearChoice(["rlObjective", "visionModel"]), true);
  assert.equal(streamingBlockersClearChoice(["maxSteps"]), true);
});

test("editing a reward leaves the reward tester's role columns as they were", () => {
  const params = {
    prompt_column: "question",
    answer_column: "answer",
    rewards: [{ name: "exact-answer", weight: 1 }],
  };
  const edited = { ...params, rewards: [{ name: "exact-answer", weight: 2 }] };
  assert.deepEqual(rewardTryColumns(edited), rewardTryColumns(params));
  assert.deepEqual(rewardTryColumns(params), {
    prompt: "question",
    answer: "answer",
  });
});

test("a cleared reward weight is not sent as 0", () => {
  assert.equal(parseRewardWeight(""), null);
  assert.equal(parseRewardWeight("  "), null);
  assert.equal(parseRewardWeight("2.5"), 2.5);
  assert.equal(parseRewardWeight("50"), 10);
  assert.equal(parseRewardWeight("-0"), -0);
});

test("YAML beta 0 is dropped for DPO and ORPO, kept for GRPO", () => {
  for (const objective of ["dpo", "orpo"]) {
    const parsed = parseYamlRlSettings(`rl:\n  objective: ${objective}\n  beta: 0\n`);
    assert.equal(parsed.rlBeta, undefined, objective);
  }
  assert.equal(
    parseYamlRlSettings("rl:\n  objective: grpo\n  beta: 0\n").rlBeta,
    0,
  );
});

test("Alpaca's output column is GRPO's answer, as in the backend", () => {
  const mapping = resolveRlMapping("grpo", ["instruction", "input", "output"], {});
  assert.deepEqual(mapping, { instruction: "prompt", output: "answer" });
});

test("a reward column named like an Object property is a column, not a role", () => {
  for (const name of ["constructor", "toString", "__proto__"]) {
    assert.equal(isRlRole(name), false, name);
  }
  assert.equal(isRlRole("answer"), true);
});
