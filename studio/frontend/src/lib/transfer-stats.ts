// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure math shared by all transfer UIs so rate/ETA semantics match; the caller owns buffer and clock. */

export type TransferSample = { t: number; b: number };

export type TransferStats = {
  rateBytesPerSecond: number;
  etaSeconds: number;
  /** Hide rate/ETA while false so the UI does not flicker absurd rates on the first tick. */
  stable: boolean;
};

export const MIN_SAMPLES = 3;
export const MIN_WINDOW_SECONDS = 3;
export const MAX_WINDOW_SECONDS = 15;
export const MAX_RETAIN_SECONDS = 60;
/** Kept regardless of age; deep enough that one outlier gap cannot become the median cadence. */
export const MIN_RETAINED_INCREASES = 8;
export const STALL_WINDOW_SECONDS = 15;
export const STALL_CADENCE_MULTIPLIER = 3;
/** Below this, an outage could set the very stall window meant to catch it. */
export const MIN_CADENCE_GAPS = 3;

/** Mutates in place; clears the buffer if the counter went backwards (cancel + restart). */
export function appendSample(
  samples: TransferSample[],
  t: number,
  b: number,
  maxWindowSeconds: number = MAX_RETAIN_SECONDS,
): TransferSample[] {
  if (samples.length > 0 && b < samples[samples.length - 1].b) {
    samples.length = 0;
  }
  const n = samples.length;
  // Collapse plateaus to first and latest reading, or a stopped transfer grows the buffer unbounded.
  if (n >= 2 && !(b > samples[n - 1].b) && !(samples[n - 1].b > samples[n - 2].b)) {
    samples[n - 1] = { t, b };
  } else {
    samples.push({ t, b });
  }
  const cutoff = t - maxWindowSeconds;
  // Not by age alone: on slow links one blob takes minutes, and measurableSpan needs the older increase.
  while (
    samples.length > 2 &&
    samples[0].t < cutoff &&
    countIncreases(samples, 1) >= MIN_RETAINED_INCREASES
  ) {
    samples.shift();
  }
  return samples;
}

function countIncreases(samples: readonly TransferSample[], from: number): number {
  let n = 0;
  for (let i = Math.max(from, 1); i < samples.length; i += 1) {
    if (samples[i].b > samples[i - 1].b) n += 1;
  }
  return n;
}

/** Tracks observed cadence: a fixed 15s calls a 60s blob cadence stalled. Uncapped on purpose. */
function stallWindowSeconds(samples: readonly TransferSample[]): number {
  const gaps: number[] = [];
  let previous = -1;
  for (let i = 1; i < samples.length; i += 1) {
    if (samples[i].b <= samples[i - 1].b) continue;
    if (previous >= 0) gaps.push(samples[i].t - samples[previous].t);
    previous = i;
  }
  if (gaps.length < MIN_CADENCE_GAPS) return STALL_WINDOW_SECONDS;
  gaps.sort((a, b) => a - b);
  // Lower median, so a long gap needs a majority before it stretches the window.
  const median = gaps[Math.floor((gaps.length - 1) / 2)];
  return Math.max(STALL_WINDOW_SECONDS, median * STALL_CADENCE_MULTIPLIER);
}

function hasRecentProgress(
  samples: readonly TransferSample[],
  last: TransferSample,
): boolean {
  const cutoff = last.t - stallWindowSeconds(samples);
  for (let index = samples.length - 2; index >= 0; index -= 1) {
    const sample = samples[index];
    if (sample.t < cutoff) break;
    if (sample.b < last.b) return true;
  }
  return false;
}

/** Prices increase-to-increase because disk-read progress arrives as plateaus and jumps (#9378). */
function measurableSpan(
  samples: readonly TransferSample[],
): [number, number] | null {
  let to = -1;
  for (let i = samples.length - 1; i > 0; i -= 1) {
    if (samples[i].b > samples[i - 1].b) {
      to = i;
      break;
    }
  }
  if (to < 1) return null;
  const stall = stallWindowSeconds(samples);
  let from = -1;
  let newer = to;
  let resumed = false;
  for (let i = to - 1; i > 0; i -= 1) {
    if (samples[i].b <= samples[i - 1].b) continue;
    // Do not reach across a stall: averaging dead time in badly underreports the resumed rate.
    if (samples[newer].t - samples[i].t > stall) {
      resumed = true;
      break;
    }
    from = i;
    newer = i;
    if (samples[to].t - samples[i].t >= MAX_WINDOW_SECONDS) break;
  }
  if (from < 1) return null;
  // Increases clumped inside the span reflect arrival jitter, so price the whole buffer instead.
  if (samples[to].t - samples[from].t < MAX_WINDOW_SECONDS) {
    if (resumed) return null;
    from = 0;
  }
  return samples[to].t - samples[from].t < MIN_WINDOW_SECONDS ? null : [from, to];
}

export function computeTransferStats(
  samples: readonly TransferSample[],
  total: number,
): TransferStats {
  const unstable = { rateBytesPerSecond: 0, etaSeconds: 0, stable: false };
  if (samples.length < MIN_SAMPLES) return unstable;
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (last.t - first.t < MIN_WINDOW_SECONDS || last.b <= first.b) {
    return unstable;
  }
  if (!hasRecentProgress(samples, last)) return unstable;
  const span = measurableSpan(samples);
  if (!span) return unstable;
  const dt = samples[span[1]].t - samples[span[0]].t;
  const db = samples[span[1]].b - samples[span[0]].b;
  const rate = dt > 0 ? db / dt : 0;
  if (!(Number.isFinite(rate) && rate > 0)) return unstable;
  const safeTotal = Number.isFinite(total) && total > 0 ? total : 0;
  const eta =
    safeTotal > 0 && last.b < safeTotal
      ? (safeTotal - last.b) / rate
      : 0;
  return { rateBytesPerSecond: rate, etaSeconds: eta, stable: true };
}
