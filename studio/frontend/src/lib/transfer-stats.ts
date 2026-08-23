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
/** Buffer depth, so sparse progress bursts still leave two points to measure. */
export const MAX_RETAIN_SECONDS = 60;
/** No byte growth for this long clears the rate instead of carrying it. */
export const STALL_WINDOW_SECONDS = 15;

/**
 * Mutate ``samples`` in place: append the sample, drop any out of the rolling
 * window, and clear the buffer if the counter went backwards (cancel + restart).
 * Returns the same array for chaining.
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
  samples.push({ t, b });
  const cutoff = t - maxWindowSeconds;
  while (samples.length > 2 && samples[0].t < cutoff) {
    samples.shift();
  }
  return samples;
}

function hasRecentProgress(
  samples: readonly TransferSample[],
  last: TransferSample,
): boolean {
  const cutoff = last.t - STALL_WINDOW_SECONDS;
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
  let from = -1;
  for (let i = to - 1; i > 0; i -= 1) {
    if (samples[i].b <= samples[i - 1].b) continue;
    from = i;
    if (samples[to].t - samples[i].t >= MAX_WINDOW_SECONDS) break;
  }
  if (from < 1) return null;
  // Increases clumped inside the smoothing span are spaced by arrival jitter, not
  // by the rate. Price the whole buffer instead, and stay silent while even that is
  // too short to divide by, so no single clump is ever published as the speed.
  if (samples[to].t - samples[from].t < MAX_WINDOW_SECONDS) from = 0;
  return samples[to].t - samples[from].t < MIN_WINDOW_SECONDS ? null : [from, to];
}

/**
 * Derive {@link TransferStats} from a window of cumulative-byte samples plus the
 * known total.
 *   * Needs ≥ {@link MIN_SAMPLES} samples spanning ≥ {@link MIN_WINDOW_SECONDS}
 *     seconds before reporting ``stable: true``.
 *   * Prices bursty cumulative-byte observations increase-to-increase.
 *   * Reports unstable after {@link STALL_WINDOW_SECONDS} without byte growth.
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
