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
test("bursts far apart still report their rate once the cadence is known", () => {
  // Silence used to be the answer here, because a fixed stall window shorter
  // than the burst period reads every healthy gap as a stall. The rate is
  // exactly recoverable though: increase-to-increase over a 90s cadence is
  // 100 MB/s on the nose, and showing it beats blanking the bar for minutes.
  const rates = burstyRates(90, 100 * MB, 600);
  assert.ok(rates.length > 0, "a 90s cadence should still publish");
  for (const rate of rates) {
    assert.ok(
      Math.abs(rate - 100 * MB) < 1 * MB,
      `published ${(rate / 1e6).toFixed(1)} MB/s, want 100`,
    );
  }
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

// The trim protects a floor of increases and a stopped transfer has none, so
// nothing was eligible to drop and a wedged download grew the buffer forever.
// 16 is what the age-only trim this replaced held.
test("a transfer that stops does not grow the sample buffer without bound", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 6 * 60 * 60; t += 1) publishedRate(samples, t, 1_000);
  assert.ok(
    samples.length <= 16,
    `buffer held ${samples.length} samples after 6h`,
  );
});

test("a long plateau keeps the span it is measured over", () => {
  // Collapsing the plateau must not lose when it began.
  const samples: TransferSample[] = [];
  publishedRate(samples, 0, 0);
  for (let t = 1; t <= 300; t += 1) publishedRate(samples, t, 500 * MB);
  assert.equal(samples[samples.length - 1].t, 300);
  assert.equal(samples[samples.length - 1].b, 500 * MB);
});

// One outlier gap (a suspend, or a browser-clamped background timer) used to be
// the only gap left in the buffer, so it became the median cadence, the stall
// window stretched to hours, and a dead transfer kept showing its last speed.
test("one outlier gap does not disable stall detection", () => {
  const samples: TransferSample[] = [];
  let bytes = 0;
  for (let t = 0; t <= 600; t += 1) {
    if (t % 5 === 0) bytes += 5 * MB;
    publishedRate(samples, t, bytes);
  }
  // 30 minutes asleep, then one burst lands, then the transfer dies.
  for (let t = 601; t <= 2_400; t += 60) publishedRate(samples, t, bytes);
  bytes += 5 * MB;
  publishedRate(samples, 2_401, bytes);
  let rate = 0;
  for (let t = 2_402; t <= 3_000; t += 1)
    rate = publishedRate(samples, t, bytes);
  assert.equal(rate, 0, "a dead transfer should stop reporting a rate");
});

// A resume keeps its byte counter, so nothing resets. Reaching back past the
// break averaged the dead time in: 20 MB/s resuming after a 30 minute drop
// published 0.64 MB/s for a full burst period. Silence is the honest answer.
test("a resume after a long stall is not priced across the stall", () => {
  const samples: TransferSample[] = [];
  let bytes = 0;
  let t = 0;
  for (; t <= 600; t += 1) {
    if (t % 60 === 0 && t > 0) bytes += 60 * 20 * MB;
    publishedRate(samples, t, bytes);
  }
  // 30 minutes with no progress and no counter reset.
  for (; t <= 600 + 1_800; t += 1) publishedRate(samples, t, bytes);
  // Then the same 20 MB/s in the same 60s bursts.
  const resumeAt = t;
  const published: number[] = [];
  for (; t <= resumeAt + 300; t += 1) {
    if ((t - resumeAt) % 60 === 0 && t > resumeAt) bytes += 60 * 20 * MB;
    const rate = publishedRate(samples, t, bytes);
    if (rate > 0) published.push(rate);
  }
  assert.ok(published.length > 0, "the resumed transfer should publish again");
  for (const rate of published) {
    assert.ok(
      Math.abs(rate - 20 * MB) < 1 * MB,
      `published ${(rate / MB).toFixed(2)} MB/s after a resume, want 20`,
    );
  }
});

// One gap is not a rhythm. When the only two increases seen straddle an outage,
// that gap used to become the cadence and size the stall window from itself, so
// the window meant to catch the outage was three times its length: a dead
// transfer published 0.28 MB/s with a 98h ETA for 90 minutes.
test("a lone outage gap is not mistaken for the burst cadence", () => {
  const samples: TransferSample[] = [];
  let bytes = 0;
  let t = 0;
  for (; t <= 5; t += 1) publishedRate(samples, t, bytes);
  bytes += 500 * MB;
  publishedRate(samples, ++t, bytes);
  for (; t <= 1_800; t += 1) publishedRate(samples, t, bytes);
  bytes += 500 * MB;
  publishedRate(samples, ++t, bytes);
  // Two increases, one gap. Progress now stops for good.
  let published = 0;
  for (const end = t + 5_400; t <= end; t += 1) {
    if (publishedRate(samples, t, bytes) > 0) published += 1;
  }
  assert.equal(
    published,
    0,
    `published a rate on ${published} ticks after death`,
  );
});

// The same trap one increase later: a normal gap followed by an outage leaves
// gaps of [5, 1800], and taking the upper of two picks the outage, so the stall
// window became 90 minutes and a dead transfer published 0.03 MB/s with a 999h
// ETA on every tick of it. Two defences: the median is not trusted below
// MIN_CADENCE_GAPS, and on an even count it takes the shorter of the middle
// pair, so an outlier needs a majority behind it.
test("a normal gap followed by an outage does not set the cadence", () => {
  const samples: TransferSample[] = [];
  let bytes = 0;
  for (let t = 0; t <= 3; t += 1) publishedRate(samples, t, bytes);
  bytes += 50 * MB;
  publishedRate(samples, 4, bytes);
  for (let t = 5; t <= 8; t += 1) publishedRate(samples, t, bytes);
  bytes += 50 * MB;
  publishedRate(samples, 9, bytes);
  for (let t = 10; t <= 1_809; t += 1) publishedRate(samples, t, bytes);
  bytes += 50 * MB;
  publishedRate(samples, 1_810, bytes);
  // Three increases, gaps of 5s and 1800s. Progress now stops for good.
  let published = 0;
  for (let t = 1_811; t <= 1_810 + 5_400; t += 1) {
    if (publishedRate(samples, t, bytes) > 0) published += 1;
  }
  assert.equal(
    published,
    0,
    `published a rate on ${published} ticks after death`,
  );
});

// Half the gaps being outages is still a minority reading of the transfer, not
// a rhythm. Four increases with gaps [5, 5, 1800, 1800] pass the count gate, so
// only the choice of middle value keeps the window off the outage.
test("outages tying with real gaps do not win the median", () => {
  const samples: TransferSample[] = [];
  let bytes = 0;
  let t = 0;
  const burst = () => {
    bytes += 50 * MB;
    publishedRate(samples, t, bytes);
  };
  for (; t <= 3; t += 1) publishedRate(samples, t, bytes);
  burst();
  for (t = 5; t <= 8; t += 1) publishedRate(samples, t, bytes);
  t = 9;
  burst();
  for (t = 10; t <= 13; t += 1) publishedRate(samples, t, bytes);
  t = 14;
  burst();
  for (t = 15; t <= 1_813; t += 1) publishedRate(samples, t, bytes);
  t = 1_814;
  burst();
  for (t = 1_815; t <= 3_613; t += 1) publishedRate(samples, t, bytes);
  t = 3_614;
  burst();
  // gaps: 5, 5, 1800, 1800. Progress now stops for good.
  let published = 0;
  for (t = 3_615; t <= 3_614 + 5_400; t += 1) {
    if (publishedRate(samples, t, bytes) > 0) published += 1;
  }
  assert.equal(
    published,
    0,
    `published a rate on ${published} ticks after death`,
  );
});

test("a restart drops the samples from the previous run", () => {
  const samples: TransferSample[] = [];
  for (let t = 0; t <= 10; t++) publishedRate(samples, t, t * 20 * MB);
  assert.equal(publishedRate(samples, 11, 0), 0);
  assert.equal(samples.length, 1);
});
