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
export const MAX_WINDOW_SECONDS = 30;
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
 * Fit one line through every sample in the rolling window instead of pricing the
 * transfer from only its two endpoints. Hub progress is observed from files on
 * disk, where sparse allocation and buffered writes can make a steady network
 * transfer appear as long plateaus followed by large byte jumps (#9378). A
 * least-squares slope makes those observation bursts contribute to the window
 * without letting a single jump become the displayed network speed.
 */
function regressionRate(samples: readonly TransferSample[]): number {
  const first = samples[0];
  const count = samples.length;
  let sumT = 0;
  let sumB = 0;
  let sumTT = 0;
  let sumTB = 0;

  for (const sample of samples) {
    const t = sample.t - first.t;
    const b = sample.b - first.b;
    sumT += t;
    sumB += b;
    sumTT += t * t;
    sumTB += t * b;
  }

  const denominator = count * sumTT - sumT * sumT;
  if (!(denominator > 0)) return 0;
  const rate = (count * sumTB - sumT * sumB) / denominator;
  return Number.isFinite(rate) && rate > 0 ? rate : 0;
}

/**
 * Derive {@link TransferStats} from a window of cumulative-byte samples plus the
 * known total.
 *   * Needs ≥ {@link MIN_SAMPLES} samples spanning ≥ {@link MIN_WINDOW_SECONDS}
 *     seconds before reporting ``stable: true``.
 *   * Smooths bursty cumulative-byte observations across the whole window.
 *   * Reports unstable after {@link STALL_WINDOW_SECONDS} without byte growth.
 *   * ETA clamps to 0 when there's no progress, no total, or total is hit.
 */
export function computeTransferStats(
  samples: readonly TransferSample[],
  total: number,
): TransferStats {
  if (samples.length < MIN_SAMPLES) {
    return { rateBytesPerSecond: 0, etaSeconds: 0, stable: false };
  }
  const first = samples[0];
  const last = samples[samples.length - 1];
  const dt = last.t - first.t;
  const db = last.b - first.b;
  if (
    dt < MIN_WINDOW_SECONDS ||
    db <= 0 ||
    (dt >= STALL_WINDOW_SECONDS && !hasRecentProgress(samples, last))
  ) {
    return { rateBytesPerSecond: 0, etaSeconds: 0, stable: false };
  }
  const rate = regressionRate(samples);
  if (!(rate > 0)) {
    return { rateBytesPerSecond: 0, etaSeconds: 0, stable: false };
  }
  const safeTotal = Number.isFinite(total) && total > 0 ? total : 0;
  const eta =
    safeTotal > 0 && last.b < safeTotal
      ? (safeTotal - last.b) / rate
      : 0;
  return { rateBytesPerSecond: rate, etaSeconds: eta, stable: true };
}
