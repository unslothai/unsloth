// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export const BLOB_DECAY_TAU_MS = 80;
export const BLOB_FLAT = 0.015;

export function decayFactor(dtMs: number, tauMs: number = BLOB_DECAY_TAU_MS): number {
  return Math.exp(-Math.max(0, dtMs) / tauMs);
}

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
