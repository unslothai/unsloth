// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { formatScore, readRewardHead, ruleTags, summarizeRule } = await import(
  "../src/features/training/lib/reward-summary.ts"
);

// Bodies as validate_rule returns them for the bundled rewards.
const strictXml = {
  type: "regex",
  pattern: "^<reasoning>\\n.*?\\n</reasoning>\\n<answer>\\n.*?\\n</answer>$",
  mode: "fullmatch",
  score: { match: 0.5, miss: 0 },
};
const exactAnswer = {
  type: "exact_match",
  extract: { between: ["<answer>", "</answer>"] },
  compare_to: "answer",
  normalize: ["strip"],
  score: { match: 2, miss: 0 },
  missing: 0,
};
const numericClose = {
  type: "numeric",
  extract: { between: ["<answer>", "</answer>"] },
  compare_to: "answer",
  bands: [
    { within: 0, score: 3 },
    { within: 0.1, score: 1.5 },
  ],
  else: -1,
  missing: -2,
};

test("formatScore signs and trims", () => {
  assert.equal(formatScore(2), "+2.0");
  assert.equal(formatScore(-1), "−1.0");
  assert.equal(formatScore(0.25), "+0.25");
  assert.equal(formatScore(0), "0");
});

test("summaries pick a key per rule type and show the hit score", () => {
  assert.deepEqual(summarizeRule(strictXml), {
    type: "regex",
    key: "regexWhole",
    params: {},
    score: "+0.5",
    otherwise: null,
  });
  const exact = summarizeRule(exactAnswer);
  assert.equal(exact.key, "exactTag");
  assert.deepEqual(exact.params, { tag: "<answer>", column: "answer" });
  assert.equal(exact.score, "+2.0");
  const numeric = summarizeRule(numericClose);
  assert.equal(numeric.key, "numericTag");
  assert.equal(numeric.score, "+3.0");
  assert.equal(numeric.otherwise, "−1.0");
  assert.equal(
    summarizeRule({
      type: "length",
      max_chars: 1200,
      score: { over: -1, under: 0 },
    }).key,
    "length",
  );
  assert.equal(summarizeRule(null).type, null);
});

test("ruleTags finds the tags a rule reads", () => {
  assert.deepEqual(ruleTags(strictXml).sort(), ["<answer>", "<reasoning>"]);
  assert.deepEqual(ruleTags(exactAnswer), ["<answer>"]);
  assert.deepEqual(ruleTags({ type: "length", max_chars: 5 }), []);
});

test("readRewardHead reads frontmatter or returns null", () => {
  assert.deepEqual(
    readRewardHead(
      "---\nname: sneaky\nkind: python\ndescription: runs code\n---\nimport os",
    ),
    { name: "sneaky", kind: "python", description: "runs code" },
  );
  assert.equal(readRewardHead("type: regex"), null);
});
