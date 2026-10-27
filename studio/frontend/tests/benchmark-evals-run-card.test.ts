// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const read = (p: string) => readFileSync(new URL(`../src/features/benchmark/${p}`, import.meta.url), "utf8");

test("the live eval card is titled by the task that is running, not the task box", () => {
  const page = read("benchmark-page.tsx");
  assert.match(page, /<BenchmarkRunPanel task=\{runTask\}/);
  assert.doesNotMatch(page, /<BenchmarkRunPanel task=\{selectedTask\}/);
  assert.match(page, /const runTask = useBenchmarkRuntimeStore\(\(s\) => s\.task\)/);
});

test("run() records its task and an attached or reset run has none", () => {
  const store = read("stores/benchmark-runtime-store.ts");
  assert.match(store, /ownsRun: true,\s*task: task \?\? "mmlu",/);
  assert.match(store, /const initialState: BenchmarkRuntimeState = \{[\s\S]*?\n  task: null,/);
});

test("history reads the run list again when a run ends, like the scoreboard", () => {
  const panel = read("components/benchmark-history-panel.tsx");
  assert.match(panel, /const phase = useBenchmarkRuntimeStore\(\(s\) => s\.phase\);/);
  assert.match(panel, /if \(phase === "running" \|\| phase === "starting"\) return;\s*fetchRuns\(\);\s*\}, \[fetchRuns, phase\]\);/);
});
