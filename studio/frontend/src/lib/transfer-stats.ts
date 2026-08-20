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
export const MAX_WINDOW_SECONDS = 15;
/**
 * Largest inter-sample gap still counted as transfer time (#9378).
 *
 * Not every consumer samples on a timer: ``useTransferStats`` appends only when
 * the byte counter changes, so bursty progress delivery leaves multi-second
 * holes between samples. A window that spans such a hole bills the post-gap
 * burst's bytes against the hole's seconds too — a stable 200 MB/s link showed
 * "3 MB/s · 5h left" right up until dense samples refilled the window and it
 * snapped to "200 MB/s · 2m". Pairs straddling a longer gap are dropped from
 * the rate's numerator AND denominator; the transfer's real pace across the
 * hole is unknowable, and guessing low is what produced the phantom ETA.
 */
export const MAX_SAMPLE_GAP_SECONDS = 5;

/**
 * Mutate ``samples`` in place: append the sample, drop any out of the rolling
 * window, and clear the buffer if the counter went backwards (cancel + restart).
 * Returns the same array for chaining.
 */
export function appendSample(
  samples: TransferSample[],
  t: number,
  b: number,
  maxWindowSeconds: number = MAX_WINDOW_SECONDS,
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

/**
 * Derive {@link TransferStats} from a window of cumulative-byte samples plus the
 * known total.
 *   * Needs ≥ {@link MIN_SAMPLES} samples spanning ≥ {@link MIN_WINDOW_SECONDS}
 *     seconds before reporting ``stable: true``.
 *   * ETA clamps to 0 when there's no progress, no total, or total is hit.
 */
export function computeTransferStats(
  samples: readonly TransferSample[],
  total: number,
): TransferStats {
  if (samples.length < MIN_SAMPLES) {
    return { rateBytesPerSecond: 0, etaSeconds: 0, stable: false };
  }
  // Accumulate the most recent run of consecutive sample pairs whose spacing
  // stays within MAX_SAMPLE_GAP_SECONDS. Walking from the newest sample means a
  // long hole ends the run exactly where the trustworthy recent data does.
  let activeDt = 0;
  let activeDb = 0;
  for (let i = samples.length - 1; i > 0; i--) {
    const dt = samples[i].t - samples[i - 1].t;
    if (dt > MAX_SAMPLE_GAP_SECONDS) {
      break;
    }
    activeDt += dt;
    activeDb += samples[i].b - samples[i - 1].b;
  }
  if (activeDt < MIN_WINDOW_SECONDS || activeDb <= 0) {
    return { rateBytesPerSecond: 0, etaSeconds: 0, stable: false };
  }
  const rate = activeDb / activeDt;
  const last = samples[samples.length - 1];
  const safeTotal = Number.isFinite(total) && total > 0 ? total : 0;
  const eta =
    safeTotal > 0 && last.b < safeTotal && rate > 0
      ? (safeTotal - last.b) / rate
      : 0;
  return { rateBytesPerSecond: rate, etaSeconds: eta, stable: true };
}
