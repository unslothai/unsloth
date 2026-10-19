// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The orb's spectrum edge settling. When the talking stops the bands decay toward
 * zero frame by frame (exponential, so it eases out) and the edge is only swapped
 * for the plain circle once it is flat; a starting voice eases the level in.
 */

/** Time constant of the release: from full to flat in ~330 ms. */
export const BLOB_DECAY_TAU_MS = 80;
/** Below this a band moves the edge by under a quarter pixel: flat. */
export const BLOB_FLAT = 0.015;

export function decayFactor(dtMs: number, tauMs: number = BLOB_DECAY_TAU_MS): number {
  return Math.exp(-Math.max(0, dtMs) / tauMs);
}

/** Decays every band in place and returns the decayed level. */
export function decayBands(
  bands: Float32Array,
  level: number,
  dtMs: number,
  tauMs: number = BLOB_DECAY_TAU_MS,
): number {
  const f = decayFactor(dtMs, tauMs);
  for (let i = 0; i < bands.length; i++) bands[i] *= f;
  return level * f;
}

export function isFlat(bands: Float32Array, level: number, eps: number = BLOB_FLAT): boolean {
  if (level > eps) return false;
  for (let i = 0; i < bands.length; i++) if (bands[i] > eps) return false;
  return true;
}

/** Frame-rate independent ease toward `target`: quicker rising than falling. */
export function easeToward(
  prev: number,
  target: number,
  dtMs: number,
  riseTauMs = 60,
  fallTauMs = 120,
): number {
  const f = decayFactor(dtMs, target > prev ? riseTauMs : fallTauMs);
  return target + (prev - target) * f;
}
