// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Pure, framework-free math behind {@link useTransferStats}.
 *
 * Split out so it can be unit-tested without a DOM/React renderer, and
 * so the training-start overlay, the chat download toast, and the
 * model-load phase UI all share the exact same rate/ETA semantics.
 *
 * No React, no `useRef`/`useState`, no timers -- the caller owns the
 * sample buffer and clock. This is the "what is the rate right now
 * given these samples" question, nothing more.
 */

export type TransferSample = { t: number; b: number };

export type TransferStats = {
  rateBytesPerSecond: number;
  etaSeconds: number;
  /**
   * False until the window has at least {@link MIN_SAMPLES} samples
   * spanning ≥ {@link MIN_WINDOW_SECONDS} with strictly forward progress.
   * Consumers should hide rate/ETA while this is false so the UI doesn't
   * flicker "123 GB/s" during the first tick.
   */
  stable: boolean;
};

export const MIN_SAMPLES = 3;
export const MIN_WINDOW_SECONDS = 3;
export const MAX_WINDOW_SECONDS = 15;

/**
 * Mutate ``samples`` in place: append the new sample, drop any that
 * fell out of the rolling window, and clear the buffer if the counter
 * went backwards (user cancelled + restarted a download, etc.).
 *
 * Returns the same array for chain-ability.
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
 * Derive {@link TransferStats} from a window of cumulative-byte samples
 * plus the known total.
 *
 *   * Needs ≥ {@link MIN_SAMPLES} samples spanning ≥ {@link MIN_WINDOW_SECONDS}
 *     seconds before it will report ``stable: true``.
 *   * ETA is clamped to 0 when: no progress, no total known, or the
 *     counter already hit the total.
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
  if (dt < MIN_WINDOW_SECONDS || db <= 0) {
    return { rateBytesPerSecond: 0, etaSeconds: 0, stable: false };
  }
  const rate = db / dt;
  const safeTotal = Number.isFinite(total) && total > 0 ? total : 0;
  const eta =
    safeTotal > 0 && last.b < safeTotal && rate > 0
      ? (safeTotal - last.b) / rate
      : 0;
  return { rateBytesPerSecond: rate, etaSeconds: eta, stable: true };
}
