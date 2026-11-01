// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// NaN and inf samples reach the UI as null. Read as 0 they made a channel look dead
// and skewed the constant check, exactly on the unstable runs these diagnostics are for.

import assert from "node:assert/strict";
import test from "node:test";

import {
  computeLayerStats,
  computeOverlaySets,
  computeViewValues,
} from "../src/features/studio/sections/activation-stats.ts";

type Rec = Parameters<typeof computeLayerStats>[0][number];

function rec(step: number, meanAbs: (number | null)[]): Rec {
  return {
    step,
    loss: null,
    layers: { "0": { mean_abs: meanAbs } },
  } as unknown as Rec;
}

// Channel 0 always missing, channel 1 alive with one missing step, channel 2 truly dead.
const RECORDS = [
  rec(1, [null, 0.5, 0.0]),
  rec(2, [null, null, 0.0]),
  rec(3, [null, 0.9, 0.0]),
];

test("an all-missing channel is neither dead nor counted", () => {
  const [layer] = computeLayerStats(RECORDS);
  assert.equal(layer.totalChannels, 2);
  assert.equal(layer.deadPct, 50);
});

test("overlays skip missing samples instead of reading them as zero", () => {
  const { dead } = computeOverlaySets(RECORDS, 2);
  assert.deepEqual([...dead], ["0:2"]);
});

test("a delta against a missing sample is missing, not a spike", () => {
  const delta = computeViewValues(RECORDS, 1, "delta");
  assert.ok(delta);
  assert.ok(Number.isNaN(delta["0"][1]));
  assert.equal(delta["0"][2], 0);
});

test("the trend fits the valid samples at their own steps", () => {
  const trend = computeViewValues(RECORDS, 2, "trend");
  assert.ok(trend);
  assert.ok(Number.isNaN(trend["0"][0]));
  assert.ok(Math.abs(trend["0"][1] - 0.2) < 1e-9);
  assert.equal(trend["0"][2], 0);
});
