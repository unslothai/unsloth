// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { evalsPickerTasks } from "../src/features/benchmark/lib/picker-tasks.ts";

test("the Evals task picker drops recipe-runner tasks and keeps lm-eval ones", () => {
  const ids = evalsPickerTasks([
    { id: "recipe_heldout", name: "Recipe held-out", runner: "recipe" },
    { id: "mmlu", name: "MMLU" },
    { id: "gsm8k", name: "GSM8K", runner: "lm_eval" },
  ]).map((t) => t.id);
  assert.deepEqual(ids, ["mmlu", "gsm8k"]);
});

test("the Evals page stores only picker tasks, so search and labels never see recipe tasks", () => {
  const page = readFileSync(
    new URL("../src/features/benchmark/benchmark-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(page, /setTasks\(evalsPickerTasks\(data\.tasks\)\)/);
  assert.doesNotMatch(page, /setTasks\(data\.tasks\)/);
});
