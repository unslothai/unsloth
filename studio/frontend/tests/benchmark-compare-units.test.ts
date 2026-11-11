// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  bestMetricScore,
  formatMetricWithUnit,
} from "../src/features/benchmark/lib/compare-metrics.ts";

const read = (f: string) =>
  readFileSync(
    new URL(`../src/features/benchmark/components/${f}`, import.meta.url),
    "utf8",
  );

test("non-ratio metrics keep their own units", () => {
  assert.equal(
    formatMetricWithUnit({ name: "acc,none", score: 0.25 }),
    "25.0%",
  );
  assert.equal(
    formatMetricWithUnit({ name: "word_perplexity,none", score: 12.3 }),
    "12.3",
  );
  assert.equal(
    formatMetricWithUnit({ name: "bleu,none", score: 34.5 }),
    "34.5",
  );
  assert.equal(
    formatMetricWithUnit({ name: "bits_per_byte,none", score: 0.7123 }),
    "0.7123",
  );
});

test("lower-is-better metrics pick the minimum as best", () => {
  assert.equal(bestMetricScore("word_perplexity,none", [12.3, 9.8, null]), 9.8);
  assert.equal(bestMetricScore("ter,none", [40, 30]), 30);
  assert.equal(bestMetricScore("acc,none", [0.2, 0.4]), 0.4);
  assert.equal(bestMetricScore("acc,none", [null]), null);
});

test("compare table and graph run list use the shared unit-aware formatter", () => {
  const compare = read("benchmark-compare-dialog.tsx");
  assert.doesNotMatch(compare, /\(score \* 100\)\.toFixed/);
  assert.match(compare, /formatMetricWithUnit\(\{\s*name: metricName/);
  assert.match(compare, /bestMetricScore\(metricName, scores\)/);
  const graph = read("benchmark-graph-dialog.tsx");
  assert.doesNotMatch(graph, /function formatMetricValue/);
  assert.match(graph, /formatMetricWithUnit\(metric\)/);
  const history = read("benchmark-history-panel.tsx");
  assert.doesNotMatch(history, /function formatMetricValue/);
  assert.doesNotMatch(history, /m\.score >= 0 && m\.score <= 1 \? "%"/);
});
