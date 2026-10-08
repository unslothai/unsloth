// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { missingRlRoles, resolveRlMapping } =
  await import("../src/features/training/lib/rl-roles.ts");

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

test("GRPO reward reference columns must come from the dataset", async () => {
  const { missingRewardColumns } =
    await import("../src/features/training/lib/rl-roles.ts");
  assert.deepEqual(missingRewardColumns(["answer"], ["question"], {}), [
    "answer",
  ]);
  assert.deepEqual(
    missingRewardColumns(["answer"], ["question", "solution"], {}),
    [],
  );
  assert.deepEqual(
    missingRewardColumns(["answer"], ["question", "gold"], { gold: "answer" }),
    [],
  );
  assert.deepEqual(
    missingRewardColumns(["unit", "answer"], ["question", "answer"], {}),
    ["unit"],
  );
});
