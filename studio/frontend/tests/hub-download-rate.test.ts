// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hub download manager derives its rate from the shared rolling-window
// estimator instead of an EMA seeded by the first sample (#7667). These pin the
// gating the progress bar relies on and the smoothing needed when backend byte
// observations arrive in disk-allocation bursts (#9378).

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

test("bursty byte observations stay close to the underlying transfer rate", () => {
  const samples: TransferSample[] = [];
  const rates: number[] = [];
  // Model a steady 100 MB/s transfer whose on-disk counter only becomes visible
  // in 1 GB allocation/write bursts every 10 seconds.
  for (let t = 0; t <= 60; t++) {
    const observedBytes = Math.floor(t / 10) * 1_000 * MB;
    const rate = publishedRate(samples, t, observedBytes);
    if (t >= 20 && rate > 0) rates.push(rate);
  }
  assert.ok(rates.length > 0);
  assert.ok(Math.min(...rates) > 75 * MB);
  assert.ok(Math.max(...rates) < 125 * MB);
});

test("a stall reports no rate instead of carrying an old window forward", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 10; t++) publishedRate(samples, t, t * 20 * MB);
  let rate = 0;
  for (let t = 11; t <= 26; t++) rate = publishedRate(samples, t, 10 * 20 * MB);
  assert.equal(rate, 0);
});

test("a restart drops the samples from the previous run", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 10; t++) publishedRate(samples, t, t * 20 * MB);
  assert.equal(publishedRate(samples, 11, 0), 0);
  assert.equal(samples.length, 1);
});
