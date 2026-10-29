// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { BenchmarkRunSummary } from "../src/features/benchmark/api/benchmark-api.ts";
import {
  ACCURACY,
  barLabel,
  comparisonGroups,
  fmtScore,
  latestScores,
  metricValue,
  scoreboardGroups,
} from "../src/features/benchmark/lib/score-bars.ts";

function run(
  id: string,
  task: string,
  model: string,
  score: number,
  stderr: string | null,
  at: string,
): BenchmarkRunSummary {
  return {
    id,
    task,
    model,
    metrics: [
      { name: "acc", score, stderr },
      { name: "acc_norm", score: score / 2, stderr: null },
    ],
    default_metric: "acc",
    n_samples: 100,
    num_fewshot: 5,
    created_at: at,
    output_path: "",
  };
}

const runs = [
  run("1", "copa", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", 0.62, "0.049", "2026-10-01T00:00:00"),
  run("2", "copa", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", 0.7, "0.04", "2026-10-02T00:00:00"),
  run("3", "copa", "org/Other-Model", 0.55, null, "2026-10-03T00:00:00"),
  run("4", "mmlu_abstract_algebra", "org/Other-Model", 0.98, "0.05", "2026-10-04T00:00:00"),
];

test("the scoreboard keeps each model's latest run per task, best first", () => {
  const { groups, models, best } = latestScores(runs, 7);
  assert.deepEqual(models, ["Llama-3.2-1B-Instruct-GGUF:Q4_K_M", "Other-Model"]);
  assert.deepEqual(
    groups.map((g) => [g.task, g.bars.map((b) => b.run.id)]),
    [
      ["mmlu_abstract_algebra", ["4"]],
      ["copa", ["2", "3"]],
    ],
  );
  assert.equal(best?.run.id, "4");
  assert.equal(latestScores(runs, 1).models.length, 1);
});

test("scoreboard rows are percent with stderr as the whisker, clamped by the chart", () => {
  const groups = scoreboardGroups(latestScores(runs, 7).groups, () => "#123456");
  const [mmlu, copa] = groups;
  assert.equal(mmlu.label, "mmlu_abstract_algebra");
  const top = mmlu.rows[0];
  assert.equal(top.value, 98);
  assert.equal(top.low, 93);
  // 103 lies past the 100 axis; BarRows' clampWhisker caps it, the row keeps the true end.
  assert.ok(Math.abs((top.high ?? 0) - 103) < 1e-9);
  assert.equal(top.tail, "± 5.0");
  assert.equal(top.color, "#123456");
  assert.equal(copa.rows[1].low, null);
  assert.equal(copa.rows[1].tail, undefined);
  assert.match(copa.rows[0].tip[1], /^70\.0% ± 4\.0 acc$/);
});

test("a bar label keeps the quant whole when the name has to be cut", () => {
  assert.equal(barLabel("Llama-3.2-1B-Instruct-GGUF:Q4_K_M"), "Llama-3.2-1B-Instruct-GGUF Q4_K_M");
  const long = barLabel(`${"x".repeat(60)}:Q8_0`);
  assert.equal(long.length, 40);
  assert.ok(long.endsWith("… Q8_0"));
  assert.equal(barLabel("y".repeat(50)).length, 40);
});

test("the comparison groups by task in run order and reads ratios as percent", () => {
  const { groups, percent, peak } = comparisonGroups(runs, ACCURACY, () => "#000", fmtScore);
  assert.equal(percent, true);
  assert.deepEqual(
    groups.map((g) => [g.label, g.rows.map((r) => r.key)]),
    [
      ["copa", ["1", "2", "3"]],
      ["mmlu_abstract_algebra", ["4"]],
    ],
  );
  assert.ok(Math.abs(peak - 103) < 1e-9);
  assert.equal(comparisonGroups(runs, "acc_norm", () => "#000", fmtScore).groups[0].rows[0].value, 31);
});

test("a metric outside 0-1 stays raw and turns the percent axis off", () => {
  const raw = run("5", "gsm8k", "m", 0.5, null, "2026-10-05T00:00:00");
  raw.metrics.push({ name: "perplexity", score: 12.5, stderr: "0.3" });
  assert.deepEqual(metricValue(raw, "perplexity"), { value: 12.5, stderr: 0.3, ratio: false });
  assert.equal(comparisonGroups([raw], "perplexity", () => "#000", String).percent, false);
  assert.equal(metricValue(raw, "missing"), null);
});
