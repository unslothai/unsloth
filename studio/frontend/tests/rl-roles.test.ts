// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { missingRlRoles, pythonRewardRow, resolveRlMapping } = await import(
  "../src/features/training/lib/rl-roles.ts"
);

test("GSM8K columns auto-map for GRPO", () => {
  assert.deepEqual(resolveRlMapping("grpo", ["question", "answer"], {}), {
    question: "prompt",
    answer: "answer",
  });
  assert.deepEqual(missingRlRoles("grpo", ["question", "answer"], {}), []);
});

test("a GRPO mapping carried over to DPO drops the answer role and reports chosen/rejected", () => {
  const mapping = { question: "prompt", answer: "answer" };
  assert.deepEqual(resolveRlMapping("dpo", ["question", "answer"], mapping), {
    question: "prompt",
  });
  assert.deepEqual(missingRlRoles("dpo", ["question", "answer"], mapping), [
    "chosen",
    "rejected",
  ]);
});

test("explicit roles win over name matching, and a stale column is ignored", () => {
  const mapping = { good: "chosen", bad: "rejected", gone: "prompt" };
  assert.deepEqual(
    resolveRlMapping("orpo", ["prompt", "chosen", "good", "bad"], mapping),
    { good: "chosen", bad: "rejected", prompt: "prompt" },
  );
});

test("a Python reward preview row keeps mapped columns and value types, as GRPO training does", () => {
  // solution is mapped to answer: training keeps it under its own name for a Python reward.
  assert.deepEqual(
    pythonRewardRow(
      {
        question: "q",
        solution: "72",
        prompt: "p",
        level: [3],
        meta: { a: 1 },
      },
      "72",
    ),
    { question: "q", solution: "72", level: [3], meta: { a: 1 }, answer: "72" },
  );
});

test("a prompt-only GRPO dataset gives a Python reward no answer", () => {
  assert.deepEqual(pythonRewardRow({ prompt: "p", id: 7 }, null), { id: 7 });
});
