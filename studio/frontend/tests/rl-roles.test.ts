// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { missingRlRoles, resolveRlMapping, syncRlMapping } =
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

test("preview and objective selector show the objective the run will use", async () => {
  const { readFileSync } = await import("node:fs");
  const read = (p: string) =>
    readFileSync(
      new URL(`../src/features/studio/${p}`, import.meta.url),
      "utf8",
    );
  assert.match(
    read("wizard/run-preview-card.tsx"),
    /trainingObjective: effectiveTrainingObjective\(s\)/,
  );
  assert.match(
    read("sections/rl/objective-select.tsx"),
    /const objective = rlLocked \? "sft" : selected;/,
  );
});

test("a start whose detected modality rules out RL cancels instead of training SFT", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/training/lib/start-fresh-training-run.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const detect = src.indexOf(
    "if (!applyDetectedDatasetModality(attempt, isImage, isAudio))",
  );
  const before = src.lastIndexOf(
    "const requestedObjective = effectiveTrainingObjective(attempt.config);",
    detect,
  );
  const cancel = src.indexOf(
    'return attempt.cancel(translate("rl.objective.modelLocked"));',
    detect,
  );
  assert.ok(before > 0 && before < detect, "objective read before detection");
  assert.ok(cancel > detect, "cancel after detection");
  assert.ok(
    cancel < src.indexOf("prepareSelectedDataset(attempt, hfToken)", detect),
  );
});

test("switching DPO to GRPO and back keeps manual preference columns", () => {
  const columns = ["q", "good", "bad", "chosen", "rejected", "answer"];
  const dpo = { q: "prompt", good: "chosen", bad: "rejected" };
  const grpo = syncRlMapping("grpo", columns, dpo);
  assert.deepEqual(resolveRlMapping("grpo", columns, grpo), {
    q: "prompt",
    answer: "answer",
  });
  const back = syncRlMapping("dpo", columns, grpo);
  assert.deepEqual(resolveRlMapping("dpo", columns, back), {
    q: "prompt",
    good: "chosen",
    bad: "rejected",
  });
  assert.equal(back.answer, "answer");
});
