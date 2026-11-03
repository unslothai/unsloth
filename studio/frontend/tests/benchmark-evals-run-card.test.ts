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
    assert.match(src, /\}, \[(fetchRuns, )?running(, runsVersion)?\]\);/, file);
    // A mid-run mount must still fetch: no early return on the running phase.
    assert.doesNotMatch(src, /if \(phase === "running" \|\| phase === "starting"\) return;/, file);
  }
});

test("metric stderr shows in the score's units, not as a raw float string", () => {
  const panel = read("components/benchmark-history-panel.tsx");
  assert.doesNotMatch(panel, /±\{m\.stderr\}/);
  assert.equal((panel.match(/±\{formatStderr\(m\)\}/g) ?? []).length, 2);
});

test("the scoreboard and comparison draw through the shared Benchmarks bar chart", () => {
  for (const file of ["components/eval-scoreboard.tsx", "components/benchmark-graph-dialog.tsx"]) {
    const src = read(file);
    assert.match(src, /import \{ BarChart \} from "@\/features\/benchmarks\/components\/bar-chart";/, file);
    assert.match(src, /<BarChart \{\.\.\.chart\} bare=\{true\} \/>/, file);
    assert.match(src, /<ChartSaveMenu svgRef=\{exportRef\}/, file);
  }
  const board = read("components/eval-scoreboard.tsx");
  assert.match(board, /<StatTile/);
  assert.doesNotMatch(board, /function Tile\(/);
  // The matplotlib route is an export now, not the in-app view.
  const graph = read("components/benchmark-graph-dialog.tsx");
  assert.doesNotMatch(graph, /<img /);
  assert.match(graph, /await downloadFile\(blob, "benchmark-chart\.png", "image\/png"\)/);
});

test("the live run log is the shared log pane, not per-line coloured divs", () => {
  const panel = read("components/benchmark-run-panel.tsx");
  assert.match(panel, /<LogPane text=\{logText\}/);
  assert.doesNotMatch(panel, /bg-black\/80|log-style/);
});

test("the scoreboard refetches when runs are deleted, and the history bumps it", () => {
  assert.match(read("components/eval-scoreboard.tsx"), /\}, \[running, runsVersion\]\);/);
  const panel = read("components/benchmark-history-panel.tsx");
  assert.match(panel, /if \(deleted\.size > 0\) bumpRunsVersion\(\);/);
  assert.match(panel, /setError\(failed\);/);
  assert.doesNotMatch(panel, /catch \{\s*\/\/ ignore/);
});

test("a run this tab started is no longer owned once it ends", () => {
  const store = read("stores/benchmark-runtime-store.ts");
  for (const phase of ["success", "canceled", "error"]) {
    assert.match(store, new RegExp(`phase: "${phase}",\\s*isRunning: false,\\s*ownsRun: false,`), phase);
  }
});

test("the Evals picker offers GGUFs only and locks while a run is on", () => {
  assert.match(read("use-evals-model.ts"), /\.filter\(\(m\) => m\.isGguf\)/);
  const picker = read("components/evals-model-picker.tsx");
  assert.match(picker, /locked=\{isRunning\}/);
  assert.doesNotMatch(picker, /loraModels/);
});

test("the run sends no HF token or model source", () => {
  for (const file of ["benchmark-page.tsx", "stores/benchmark-runtime-store.ts", "api/benchmark-api.ts"]) {
    assert.doesNotMatch(read(file), /prepareHfTokenForUse|hf_token|model_source/, file);
  }
});

test("task types are labelled from the values the backend sends", () => {
  const page = read("benchmark-page.tsx");
  assert.match(page, /tl === "log_likelihood"/);
  assert.match(page, /tl === "generation"/);
  assert.doesNotMatch(page, /greedy_until/);
});

test("the graph dialog derives its run order instead of setting state in a memo", () => {
  const dialog = read("components/benchmark-graph-dialog.tsx");
  assert.doesNotMatch(dialog, /useMemo\(\(\) => \{\s*set[A-Z]/);
  assert.doesNotMatch(dialog, /benchmark\.graph\.accuracy/);
});
