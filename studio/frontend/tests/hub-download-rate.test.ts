// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hub download manager rates its progress bar through the shared estimator,
// not an EMA seeded by the first sample (#7667). These pin the gating it relies
// on, and the smoothing for byte counts arriving in disk bursts (#9378).

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

/** Steady ``rate`` whose observed counter only advances every ``burst`` seconds. */
function burstyRates(burst: number, rate: number, until = 300): number[] {
  const samples: TransferSample[] = [];
  const published: number[] = [];
  for (let t = 0; t <= until; t++) {
    const observed = Math.floor(t / burst) * burst * rate;
    const seen = publishedRate(samples, t, observed);
    if (t >= 3 * burst && seen > 0) published.push(seen);
  }
  return published;
}

// Sparse allocation turns a steady transfer into plateaus and jumps, so pricing
// the window endpoints swung between "very slow, hours left" and hundreds of MB/s.
test("bursty byte observations report the underlying transfer rate", () => {
  for (const burst of [2, 5, 10, 15, 20, 30, 45]) {
    const rates = burstyRates(burst, 100 * MB);
    assert.ok(rates.length > 0, `no rate published for ${burst}s bursts`);
    for (const rate of rates) {
      assert.ok(
        Math.abs(rate - 100 * MB) < 5 * MB,
        `${burst}s bursts published ${(rate / MB).toFixed(1)} MB/s`,
      );
    }
  }
});

// Two jumps are the minimum that carries timing; below that the honest answer is
// no rate, not a number set by where the window happened to cut.
test("bursts too far apart to measure publish no rate at all", () => {
  assert.equal(burstyRates(90, 100 * MB, 600).length, 0);
});

// External job callbacks can fire back to back, so the first two increases may be
// milliseconds apart while the buffer is still short. Dividing by that gap is how
// the gate used to leak a "123 GB/s" first tick.
test("increases arriving together during warm-up are not a rate", () => {
  const samples: TransferSample[] = [];
  publishedRate(samples, 0, 0);
  publishedRate(samples, 3, 0);
  publishedRate(samples, 3.01, 1_000 * MB);
  const rate = publishedRate(samples, 3.02, 2_000 * MB);
  // 2 GB over the 3.02s actually observed, not 1 GB over the 10ms between them.
  assert.ok(rate < 1_000 * MB, `published ${(rate / MB).toFixed(0)} MB/s`);
});

// Recovering from a stall longer than the buffer leaves only the clump that just
// landed. Its samples are a second apart, so measuring across them alone would
// turn one xorb into the transfer speed.
test("a clump landing after a long stall does not become the speed", () => {
  const samples: TransferSample[] = [];
  let bytes = 0;
  let peak = 0;
  for (let t = 0; t <= 95; t++) {
    if (t <= 20) bytes = t * 100 * MB;
    else if (t === 82 || t === 83) bytes += 1_000 * MB;
    peak = Math.max(peak, publishedRate(samples, t, bytes));
  }
  assert.ok(peak <= 100 * MB, `published ${(peak / MB).toFixed(0)} MB/s`);
});

test("a stall reports no rate instead of carrying an old window forward", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 10; t++) publishedRate(samples, t, t * 20 * MB);
  let rate = 0;
  for (let t = 11; t <= 26; t++) rate = publishedRate(samples, t, 10 * 20 * MB);
  assert.equal(rate, 0);
});

// The chat toast, model-load UI and training overlay share this on dense feeds,
// so the burst handling must not slow their reaction down.
test("a dense feed still tracks a rate change within the smoothing span", () => {
  const samples: TransferSample[] = [];
  let bytes = 0;
  let rate = 0;
  for (let t = 0; t <= 40; t++) {
    bytes += (t < 20 ? 200 : 50) * MB;
    rate = publishedRate(samples, t, bytes);
  }
  assert.ok(Math.abs(rate - 50 * MB) < 1);
});

test("a restart drops the samples from the previous run", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 10; t++) publishedRate(samples, t, t * 20 * MB);
  assert.equal(publishedRate(samples, 11, 0), 0);
  assert.equal(samples.length, 1);
});
