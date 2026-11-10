// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import yaml from "js-yaml";
import type { RewardRecord } from "../src/features/training/api/rewards-api.ts";
import type { DraftRule } from "../src/features/training/lib/reward-editor.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  copyRewardName,
  defaultRewardRule,
  normalizeRewardName,
  rewardDraftMarkdown,
} = await import("../src/features/training/lib/reward-editor.ts");
const { canEditReward, RULE_TYPES } = await import(
  "../src/features/training/lib/reward-summary.ts"
);

function record(
  name: string,
  rule: DraftRule | null,
  source: "user" | "bundled" = "user",
): RewardRecord {
  return {
    name,
    rule,
    source,
    kind: "rule",
    description: "",
    valid: true,
    shadowed: false,
    error: null,
  };
}

test("a draft round-trips through YAML with typed numbers, Unicode and multiline text", () => {
  const rule = {
    type: "numeric",
    compare_to: "target",
    extract: { regex: "([0-9.,]+)" },
    bands: [
      { within: "0", score: "3" },
      { within: "0.1", score: 1.5 },
    ],
    else: "-1",
    missing: "-2.5",
  } as DraftRule;
  const description = "Line one: # note\nשלום\n---\nkind: python";
  const [head, body] = yaml.loadAll(
    rewardDraftMarkdown({ name: " Test-Reward ", description, rule }),
  );
  assert.deepEqual(head, { name: "test-reward", kind: "rule", description });
  assert.deepEqual(body, {
    ...rule,
    bands: [
      { within: 0, score: 3 },
      { within: 0.1, score: 1.5 },
    ],
    else: -1,
    missing: -2.5,
  });
});

test("unparseable numbers reach the backend as typed, so its validation names the field", () => {
  const rule = {
    type: "length",
    max_chars: "",
    score: { under: "x", over: -1 },
  };
  const [, body] = yaml.loadAll(
    rewardDraftMarkdown({
      name: "n",
      description: "",
      rule: rule as DraftRule,
    }),
  );
  assert.deepEqual(body, rule);
});

test("every rule type has a default, and bundled presets seed theirs", () => {
  for (const type of RULE_TYPES) {
    assert.equal(defaultRewardRule(type, []).type, type);
  }
  const preset = { type: "exact_match", compare_to: "answer", normalize: [] };
  const bundled = record("exact-answer", preset as DraftRule, "bundled");
  const seeded = defaultRewardRule("exact_match", [bundled]);
  assert.deepEqual(seeded, preset);
  assert.notEqual(seeded, preset, "the preset is copied, not shared");
  const fallback = defaultRewardRule("exact_match", [
    { ...bundled, source: "user" },
  ]);
  assert.deepEqual(fallback.normalize, ["strip"]);
});

test("copies get a free name within the 64 character limit", () => {
  const long = "a".repeat(64);
  const library = [record(`${long.slice(0, 52)}-copy`, null)];
  const name = copyRewardName(long, library);
  assert.equal(name, `${long.slice(0, 52)}-copy-2`);
  assert.ok(normalizeRewardName(name));
});

test("names follow the backend rule", () => {
  assert.equal(normalizeRewardName(" My-Reward "), "my-reward");
  for (const bad of ["", "-lead", "under_score", "a".repeat(65), "sp ace"]) {
    assert.equal(normalizeRewardName(bad), null, bad);
  }
});

test("only valid, active rule rewards of a known type open in the editor", () => {
  const ok = record("x", { type: "regex" } as DraftRule);
  assert.equal(canEditReward(ok), true);
  assert.equal(canEditReward({ ...ok, valid: false }), false);
  assert.equal(canEditReward({ ...ok, shadowed: true }), false);
  assert.equal(canEditReward({ ...ok, kind: "python" }), false);
  assert.equal(canEditReward({ ...ok, rule: { type: "custom" } }), false);
  assert.equal(canEditReward(undefined), false);
});
