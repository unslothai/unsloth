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
  canEditReward,
  copyRewardName,
  defaultRewardRule,
  groupRewardPreviews,
  rewardDraftMarkdown,
  rewardReference,
  RewardDraftError,
} = await import("../src/features/training/lib/reward-editor.ts");

function record(
  name: string,
  rule: DraftRule,
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
const exact = defaultRewardRule("exact_match", []);
const rules: DraftRule[] = [
  {
    ...exact,
    normalize: ["lower", "strip", "collapse_spaces", "remove_commas"],
    missing: -2,
  },
  {
    type: "numeric",
    compare_to: "target",
    extract: { regex: "(?P<value>[0-9.,]+)" },
    reference_extract: null,
    missing: -3,
    bands: [
      { within: 0, score: 3 },
      { within: 0.1, score: 1.5 },
    ],
    else: -1,
  },
  {
    type: "regex",
    pattern: "^<answer>\\n.*?\\n</answer>$",
    mode: "search",
    score: { match: 0.5, miss: -0.5 },
  },
  {
    type: "json_schema",
    extract: { between: ["```json\n", "\n```"] },
    schema: { type: "object", required: ["answer", "a,b", "שלום\nעולם", ""] },
    score: { match: 1, miss: 0 },
  },
  { type: "length", max_chars: 100, score: { under: 0.1, over: -1 } },
];
for (const rule of rules) {
  test(`${rule.type} preserves fields, ordering, Unicode and multiline text through YAML`, () => {
    const draft = {
      name: "test-reward",
      description: "Line one: # note\nשלום\n---\nkind: python",
      rule,
    };
    const before = structuredClone(draft);
    const [metadata, body] = yaml.loadAll(rewardDraftMarkdown(draft));
    assert.deepEqual(metadata, {
      name: draft.name,
      kind: "rule",
      description: draft.description,
    });
    assert.deepEqual(body, rule);
    assert.deepEqual(draft, before);
  });
}

test("normalizes names and rejects invalid identifiers", () => {
  const [metadata] = yaml.loadAll(
    rewardDraftMarkdown({
      name: " Ｍｙ-Ｒｅｗａｒｄ ",
      description: "",
      rule: exact,
    }),
  );
  assert.equal((metadata as { name: string }).name, "my-reward");
  for (const name of [
    "",
    "a".repeat(65),
    "../escape",
    "-first",
    "has space",
    "שם",
  ]) {
    assert.throws(
      () => rewardDraftMarkdown({ name, description: "", rule: exact }),
      (e) => e instanceof RewardDraftError && e.code === "invalidName",
    );
  }
});

test("number inputs preserve zero and negative scores, rejecting empty and nonfinite values", () => {
  const draft = {
    name: "numbers",
    description: "",
    rule: {
      type: "length",
      max_chars: "0",
      score: { over: "-1.25", under: "0" },
    } as DraftRule,
  };
  assert.deepEqual(yaml.loadAll(rewardDraftMarkdown(draft))[1], {
    type: "length",
    max_chars: 0,
    score: { over: -1.25, under: 0 },
  });
  for (const max_chars of ["", " ", "Infinity", "NaN", "1.5"]) {
    assert.throws(
      () =>
        rewardDraftMarkdown({ ...draft, rule: { ...draft.rule, max_chars } }),
      (e) => e instanceof RewardDraftError && e.code === "invalidNumber",
    );
  }
  assert.throws(
    () =>
      rewardDraftMarkdown({
        ...draft,
        rule: { ...draft.rule, score: { over: "", under: 0 } },
      }),
    RewardDraftError,
  );
});

test("incomplete conditions cannot be serialized and UTF-8 size is bounded", () => {
  const incomplete: DraftRule[] = [
    { ...exact, compare_to: "" },
    { ...exact, extract: { between: ["start", ""] } },
    { ...exact, reference_extract: { regex: "" } },
    { type: "numeric", compare_to: "answer", bands: [], else: 0, missing: 0 },
    { type: "regex", pattern: "", score: { match: 1, miss: 0 } },
  ];
  for (const rule of incomplete) {
    assert.throws(
      () => rewardDraftMarkdown({ name: "incomplete", description: "", rule }),
      RewardDraftError,
    );
  }
  assert.throws(
    () =>
      rewardDraftMarkdown({
        name: "large",
        description: "界".repeat(22000),
        rule: exact,
      }),
    (e) => e instanceof RewardDraftError && e.code === "tooLarge",
  );
});

test("customizing a bundled default never mutates the library", () => {
  const preset = record("exact-answer", exact, "bundled");
  const draft = defaultRewardRule("exact_match", [preset]);
  draft.normalize?.reverse();
  draft.score!.match = 99;
  assert.deepEqual(preset.rule, exact);
  assert.notDeepEqual(draft, exact);
});

test("copy names avoid user and bundled collisions within the name limit", () => {
  assert.equal(
    copyRewardName("answer", [
      record("answer-copy", exact),
      record("answer-copy-2", exact, "bundled"),
    ]),
    "answer-copy-3",
  );
  assert.ok(copyRewardName("a".repeat(64), []).length <= 64);
});

test("only valid supported rule records can be edited", () => {
  const supported = record("ok", exact);
  assert.equal(canEditReward(supported), true);
  for (const patch of [
    { kind: "python" as const },
    { valid: false },
    { shadowed: true },
    { rule: null },
    { rule: { type: "future" } },
  ]) {
    assert.equal(canEditReward({ ...supported, ...patch }), false);
  }
});

test("references follow the answer mapping and keep custom columns independent", () => {
  const row = {
    question: "?",
    answer: "old",
    solution: "72",
    alternate_answer: "73",
  };
  assert.deepEqual(rewardReference("answer", row, { solution: "answer" }), {
    source: "solution",
    value: "72",
  });
  assert.deepEqual(
    rewardReference("alternate_answer", row, { solution: "answer" }),
    { source: "alternate_answer", value: "73" },
  );
  assert.deepEqual(rewardReference("absent", row, {}), {
    source: null,
    value: "",
  });
  assert.deepEqual(rewardReference("answer", null, {}), {
    source: null,
    value: "",
  });
});

test("auto-mapping follows backend alias priority independent of column order", () => {
  assert.deepEqual(
    rewardReference(
      "answer",
      { solution: "wrong", answer: "right", question: "?" },
      {},
    ),
    { source: "answer", value: "right" },
  );
  assert.deepEqual(
    rewardReference(
      "answer",
      { label: "wrong", solution: "right", question: "?" },
      {},
    ),
    { source: "solution", value: "right" },
  );
});

test("non-text cells stay manual instead of pretending JSON equals Python str", () => {
  for (const value of [null, true, 1.0, ["a"], { answer: 72 }]) {
    assert.deepEqual(rewardReference("answer", { answer: value }, {}), {
      source: null,
      value: "",
    });
  }
  assert.deepEqual(rewardReference("answer", { answer: "" }, {}), {
    source: "answer",
    value: "",
  });
});

test("mixed-column preview groups preserve distinct references, weights and missing values", () => {
  const library = [
    record("correct", exact),
    record("alternate", { ...exact, compare_to: "alternate_answer" }),
    record("format", rules[2]),
    record("same-column", exact),
  ];
  const selections = [
    { name: "correct", weight: 2 },
    { name: "alternate", weight: -1 },
    { name: "format", weight: 0 },
    { name: "same-column", weight: 1 },
  ];
  assert.deepEqual(
    groupRewardPreviews(selections, library, {
      answer: "72",
      alternate_answer: "73",
    }),
    [
      { rewards: [selections[0], selections[3]], reference: "72" },
      { rewards: [selections[1]], reference: "73" },
      { rewards: [selections[2]], reference: null },
    ],
  );
  assert.equal(
    groupRewardPreviews([selections[1]], library, {})[0].reference,
    null,
  );
  assert.equal(
    groupRewardPreviews([selections[1]], library, { alternate_answer: "" })[0]
      .reference,
    "",
  );
});
