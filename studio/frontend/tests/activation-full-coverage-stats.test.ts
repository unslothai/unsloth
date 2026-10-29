// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// With full coverage the backend counts dead, constant and outlier channels over every
// hidden channel, so the panels must read those counts instead of the 64 sampled columns.

import assert from "node:assert/strict";
import test from "node:test";

import { computeLayerStats, computeTrendData } from "../src/features/studio/sections/activation-stats.ts";
import { readSrc } from "./helpers/kit.ts";

const sampled = (v: number) => ({ mean_abs: [v, v, v, v], mean: [v, v, v, v] });

test("full-coverage counts replace the sampled estimate", () => {
  const records = [
    { step: 0, layers: { "0": sampled(1) } },
    {
      step: 3,
      layers: { "0": sampled(1) },
      coverage: {
        "0": { channels: 1024, dead: 256, constant: 512, outliers: 3, top_outliers: [[35, 37.5], [7, 9.1], [1, 6.2]] },
      },
    },
  ];
  const [layer] = computeLayerStats(records as never);
  assert.equal(layer.totalChannels, 1024);
  assert.equal(layer.deadPct, 25);
  assert.equal(layer.constantPct, 50);
  assert.equal(layer.fullCoverage, true);
  assert.equal(layer.outliers, 3);
  assert.deepEqual(layer.topOutlier, { channel: 35, ratio: 37.5 });

  // Step 0 has no coverage yet, so its trend point falls back to the samples (all alive).
  const trend = computeTrendData(records as never);
  assert.equal(trend[0].deadPct, 0);
  assert.equal(trend[1].deadPct, 25);
});

test("a run without coverage keeps the sampled counts", () => {
  const records = [{ step: 0, layers: { "0": { mean_abs: [0, 1, 1, 1], mean: [0, 1, 1, 1] } } }];
  const [layer] = computeLayerStats(records as never);
  assert.equal(layer.totalChannels, 4);
  assert.equal(layer.deadPct, 25);
  assert.equal(layer.fullCoverage, false);
  assert.equal(layer.topOutlier, null);
});

test("a null outlier ratio is skipped, not shown", () => {
  const records = [
    {
      step: 0,
      layers: { "0": sampled(1) },
      coverage: { "0": { channels: 8, dead: 0, constant: 0, outliers: 1, top_outliers: [[2, null], [5, 7]] } },
    },
  ];
  assert.deepEqual(computeLayerStats(records as never)[0].topOutlier, { channel: 5, ratio: 7 });
});

test("Neuron Health uses the shared stats and the heatmap marks pinned columns", () => {
  const health = readSrc("features/studio/sections/dead-neuron-panel.tsx");
  assert.match(health, /import \{ computeLayerStats, type LayerStats \} from "\.\/activation-stats";/);
  assert.doesNotMatch(health, /function computeLayerStats/);
  const heatmap = readSrc("features/studio/sections/neuron-heatmap-section.tsx");
  assert.match(heatmap, /new Set\(metadata\?\.pinned_channels \?\? \[\]\)/);
  assert.match(heatmap, /if \(!pinnedChannels\.has\(ch\)\) return;/);
});

test("a poll tick in flight when training ends does not reschedule", () => {
  const hook = readSrc("features/training/hooks/use-activation-data.ts");
  assert.match(hook, /if \(mountedRef\.current && !cancelled\) \{\s*schedule\(\);/);
  assert.match(hook, /return \(\) => \{\s*cancelled = true;/);
});
