// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { benchmarkView, percent } = await import(
  "../src/features/flows/editor/benchmark-status.ts"
);
const { readable } = await import("../src/features/flows/editor/value-kinds.ts");

const catalog = JSON.parse(
  readFileSync(new URL("./fixtures/flows-catalog.json", import.meta.url), "utf8"),
);
const node = catalog.nodes.find(
  (n: { type: string }) => n.type === "train.benchmark",
);

const base = {
  status: "idle" as const,
  params: { task: "Recipe held-out" },
  hasRows: true,
  outputs: null,
  phase: null,
  error: null,
};

test("Benchmark keeps task, scorer, rows and model up front and scores into three outputs", () => {
  const simple = node.params
    .filter((p: { advanced?: boolean }) => !p.advanced)
    .map((p: { id: string }) => p.id);
  assert.deepEqual(simple, ["task", "scorer", "max_rows", "model"]);
  assert.deepEqual(
    node.outputs.map((p: { id: string; type: string }) => `${p.id}:${p.type}`),
    ["score:number", "results:dataset", "failures:dataset"],
  );
  const inputs = node.inputs.filter((p: { param?: string }) => !p.param);
  assert.deepEqual(
    inputs.map((p: { id: string; type: string; optional: boolean }) => [p.id, p.type, p.optional]),
    [
      ["checkpoint", "checkpoint", true],
      ["model", "model", true],
      ["dataset", "dataset", true],
    ],
  );
});

test("the card reads its state from the run", () => {
  assert.equal(benchmarkView({ ...base, hasRows: false }).state, "needs_rows");
  assert.equal(
    benchmarkView({ ...base, hasRows: false, params: { task: "HellaSwag" } }).state,
    "ready",
  );
  const scoring = benchmarkView({ ...base, status: "running", phase: "Scoring 20 rows" });
  assert.deepEqual([scoring.state, scoring.detail], ["scoring", "Scoring 20 rows"]);
  assert.equal(benchmarkView({ ...base, status: "failed", error: "x" }).state, "failed");
  const scored = benchmarkView({
    ...base,
    status: "done",
    outputs: {
      score: 100,
      results: { kind: "dataset", passed: 2, scored: 2, base_score: 0, metric: "reference match", run_id: "r1" },
    },
  });
  assert.equal(scored.state, "scored");
  assert.deepEqual(scored.result, {
    score: 100,
    passed: 2,
    scored: 2,
    baseScore: 0,
    metric: "reference match",
    runId: "r1",
  });
  assert.equal(percent(25.6), "26%");
  assert.equal(percent(null), "-");
});

test("a judge cell reads as key: value lines, not raw JSON", () => {
  assert.equal(
    readable({ correctness: { score: 4, reasoning: "Same rows" }, passed: true }),
    "correctness: score: 4, reasoning: Same rows\npassed: true",
  );
  assert.equal(readable(["a", 1]), "a, 1");
});
