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

test("history and scoreboard load on mount, mid-run included, and again when a run ends", () => {
  for (const file of ["components/benchmark-history-panel.tsx", "components/eval-scoreboard.tsx"]) {
    const src = read(file);
    assert.match(src, /\(s\) => s\.phase === "running" \|\| s\.phase === "starting",/, file);
    assert.match(src, /\}, \[(fetchRuns, )?running\]\);/, file);
    // A mid-run mount must still fetch: no early return on the running phase.
    assert.doesNotMatch(src, /if \(phase === "running" \|\| phase === "starting"\) return;/, file);
  }
});

test("metric stderr shows in the score's units, not as a raw float string", () => {
  const panel = read("components/benchmark-history-panel.tsx");
  assert.doesNotMatch(panel, /±\{m\.stderr\}/);
  assert.equal((panel.match(/±\{formatStderr\(m\)\}/g) ?? []).length, 2);
});

test("a scoreboard row keeps the quant visible when the model name truncates", () => {
  const board = read("components/eval-scoreboard.tsx");
  assert.match(board, /<ModelLabel model=\{b\.model\} \/>/);
  assert.match(board, /<span className="truncate">\{name\}<\/span>\s*\{quant && <span className="shrink-0/);
});
