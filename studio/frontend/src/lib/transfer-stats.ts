// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Pure, framework-free math behind {@link useTransferStats}.
 *
 * Split out for unit-testing without React, and so the training-start overlay,
 * chat download toast, model-load UI and hub download manager share identical
 * rate/ETA semantics. No React/timers -- the caller owns the buffer and clock.
 */

export type TransferSample = { t: number; b: number };

export type TransferStats = {
  rateBytesPerSecond: number;
  etaSeconds: number;
  /**
   * False until the window has ≥ {@link MIN_SAMPLES} samples spanning ≥
   * {@link MIN_WINDOW_SECONDS} with forward progress. Hide rate/ETA while false
   * so the UI doesn't flicker "123 GB/s" during the first tick.
   */
  stable: boolean;
};

export const MIN_SAMPLES = 3;
export const MIN_WINDOW_SECONDS = 3;
/** Span the rate is averaged over once progress is dense enough to fill it. */
export const MAX_WINDOW_SECONDS = 15;
/** Buffer depth by age; {@link MIN_RETAINED_INCREASES} overrides it when sparse. */
export const MAX_RETAIN_SECONDS = 60;
/**
 * Increases kept regardless of age, so a slow cadence still leaves a span. Deep
 * enough that one outlier gap cannot become the median cadence on its own.
 */
export const MIN_RETAINED_INCREASES = 8;
/** Floor on the no-progress timeout, before any cadence has been observed. */
export const STALL_WINDOW_SECONDS = 15;
/** A silence this many times the observed increase cadence reads as a stall. */
export const STALL_CADENCE_MULTIPLIER = 3;
/**
 * Gaps needed before the median is trusted as the cadence. Below this an outage
 * can be most of what you have seen, and sizing the stall window from it is
 * circular: the outage would set the window meant to catch it.
 */
export const MIN_CADENCE_GAPS = 3;

/**
 * Mutate ``samples`` in place: append the sample, drop those out of the rolling
 * window while {@link MIN_RETAINED_INCREASES} survive, and clear the buffer if
 * the counter went backwards (cancel + restart). Returns it for chaining.
 */
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
  // Collapse a plateau to its first and latest reading. Nothing here reads the
  // ones between, and the trim below cannot drop them (too few increases left
  // to protect), so a stopped transfer would grow the buffer without bound.
  if (n >= 2 && !(b > samples[n - 1].b) && !(samples[n - 1].b > samples[n - 2].b)) {
    samples[n - 1] = { t, b };
  } else {
    samples.push({ t, b });
  }
  const cutoff = t - maxWindowSeconds;
  // Age alone cannot decide this: on a slow link one blob takes minutes, so
  // trimming by seconds drops the older increase measurableSpan needs.
  while (
    samples.length > 2 &&
    samples[0].t < cutoff &&
    countIncreases(samples, 1) >= MIN_RETAINED_INCREASES
  ) {
    samples.shift();
  }
  return samples;
}

/** Byte increases in ``samples`` at or after ``from``. */
function countIncreases(samples: readonly TransferSample[], from: number): number {
  let n = 0;
  for (let i = Math.max(from, 1); i < samples.length; i += 1) {
    if (samples[i].b > samples[i - 1].b) n += 1;
  }
  return n;
}

/**
 * How long a silence runs before it is a stall rather than a gap between jumps.
 * A fixed 15s calls a 60s blob cadence stalled every time, blanking 95% of
 * ticks at 3 MB/s, so track the observed cadence. Uncapped: any cap is a cliff
 * for a repo slower than it.
 */
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
  // Lower median: on an even count take the shorter of the middle pair, so a
  // long gap needs a majority behind it before it can stretch the window.
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

/**
 * The pair to price the transfer from: newest byte increase, and the tightest
 * earlier increase ≥ {@link MAX_WINDOW_SECONDS} back (else the oldest held).
 *
 * Hub progress is read off disk, where sparse allocation makes a steady transfer
 * look like plateaus and jumps (#9378). Increase-to-increase spans a whole number
 * of jumps, so the partial plateau at each end stops tilting the rate; a dense
 * feed increases every sample and so still averages over the full span.
 *
 * ``null`` below two increases: one jump alone carries no timing to measure.
 */
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
    // A silence past the stall window is a break, not a slow burst. Reaching
    // across it averages the dead time in: 20 MB/s resuming after a 30 minute
    // drop published 0.64 MB/s.
    if (samples[newer].t - samples[i].t > stall) {
      resumed = true;
      break;
    }
    from = i;
    newer = i;
    if (samples[to].t - samples[i].t >= MAX_WINDOW_SECONDS) break;
  }
  if (from < 1) return null;
  // Increases clumped inside the smoothing span are spaced by arrival jitter, not
  // by the rate. Price the whole buffer instead, and stay silent while even that is
  // too short to divide by, so no single clump is ever published as the speed.
  if (samples[to].t - samples[from].t < MAX_WINDOW_SECONDS) {
    // Unless that would reach back over a break: stay silent until the resumed
    // transfer has a span of its own.
    if (resumed) return null;
    from = 0;
  }
  return samples[to].t - samples[from].t < MIN_WINDOW_SECONDS ? null : [from, to];
}

/**
 * Derive {@link TransferStats} from a window of cumulative-byte samples plus the
 * known total.
 *   * Needs ≥ {@link MIN_SAMPLES} samples spanning ≥ {@link MIN_WINDOW_SECONDS}
 *     seconds before reporting ``stable: true``.
 *   * Prices bursty cumulative-byte observations increase-to-increase.
 *   * Reports unstable once a silence outruns the observed increase cadence.
 *   * ETA clamps to 0 when there's no progress, no total, or total is hit.
 */
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
