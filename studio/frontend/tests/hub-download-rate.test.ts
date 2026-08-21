// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hub download manager now derives its rate from the shared rolling-window
// estimator instead of an EMA seeded by the first sample (#7667). These pin the
// gating the progress bar relies on: no rate (and so no ETA) until the window is
// trustworthy, and no tiny positive rate while a transfer is stalled.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type TransferSample,
  appendSample,
  computeTransferStats,
} from "../src/lib/transfer-stats.ts";

const TOTAL = 6.8e9;
const MB = 1e6;

/** Mirrors poll-loop's applySpeedSample: the rate published to the UI. */
function publishedRate(
  samples: TransferSample[],
  t: number,
  b: number,
): number {
  appendSample(samples, t, b);
  const stats = computeTransferStats(samples, TOTAL);
  return stats.stable ? stats.rateBytesPerSecond : 0;
}

test("connection ramp-up publishes no rate until the window is trustworthy", () => {
  const samples: TransferSample[] = [];
  assert.equal(publishedRate(samples, 0, 0), 0);
  assert.equal(publishedRate(samples, 1, 100), 0);
  assert.equal(publishedRate(samples, 2, 200), 0);
  // Only now are there 3 samples spanning 3s of forward progress.
  assert.ok(publishedRate(samples, 3, 30 * MB) > 0);
});

test("a steady transfer reports its true rate", () => {
  const samples: TransferSample[] = [];
  let rate = 0;
  for (let t = 0; t <= 10; t++) rate = publishedRate(samples, t, t * 20 * MB);
  assert.ok(Math.abs(rate - 20 * MB) < 1);
});

test("a stall reports no rate instead of decaying toward zero", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 10; t++) publishedRate(samples, t, t * 20 * MB);
  let rate = 0;
  for (let t = 11; t <= 40; t++) rate = publishedRate(samples, t, 10 * 20 * MB);
  assert.equal(rate, 0);
});

test("a restart drops the samples from the previous run", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 10; t++) publishedRate(samples, t, t * 20 * MB);
  assert.equal(publishedRate(samples, 11, 0), 0);
  assert.equal(samples.length, 1);
});

test("a delivery gap is not billed as slow transfer time (#9378)", () => {
  // Bursty delivery: progress events pause for ~8s, then resume at 200 MB/s.
  // A window that spans the hole bills the post-hole bytes against the hole's
  // seconds too, so a stable link reads as a fraction of its pace (the "5h
  // left, then 2m left" flip). The honest early answer is "no trustworthy rate
  // yet"; the steady answer is the hole-free recent pace.
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 5; t++) publishedRate(samples, t, t * 20 * MB);
  // Reporting hole t=6..12; delivery resumes as 200 MB/s events at t=13+.
  // Before 3s of gap-free samples: no rate (the old window would already be
  // billing the resume bytes against the hole here).
  assert.equal(publishedRate(samples, 13, 5 * 20 * MB + 200 * MB), 0);
  assert.equal(publishedRate(samples, 14, 5 * 20 * MB + 400 * MB), 0);
  assert.equal(publishedRate(samples, 15, 5 * 20 * MB + 600 * MB), 0);
  // Three gap-free seconds accumulated: the window reports the true recent
  // pace, hole excluded — not bytes/(hole + pace).
  const rate = publishedRate(samples, 16, 5 * 20 * MB + 800 * MB);
  assert.ok(Math.abs(rate - 200 * MB) < 1, `rate was ${rate}`);
  const steady = publishedRate(samples, 17, 5 * 20 * MB + 1000 * MB);
  assert.ok(Math.abs(steady - 200 * MB) < 1, `rate was ${steady}`);
});

test("a sparse but gap-free feed still reports its rate", () => {
  // Change-driven consumers can legitimately sample every 4s; as long as no
  // pair exceeds the gap tolerance the rate must stay trustworthy.
  const samples: TransferSample[] = [];
  let rate = 0;
  for (let t = 0; t <= 20; t += 4) rate = publishedRate(samples, t, (t / 4) * 100 * MB);
  assert.ok(Math.abs(rate - 25 * MB) < 1, `rate was ${rate}`);
});
