// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The orb's spectrum edge as three lobes. Each lobe sweeps the bands low to high
 * under a cosine envelope that is zero between lobes, so the seam where one lobe's
 * high bands meet the next one's low bands never shows. Half of each lobe is the
 * loudest band, so a voice whose energy sits off-centre still reads as three lobes,
 * and the result is scaled so the furthest point is exactly the loudest band, the
 * same reach the mirrored two-hump edge had.
 *
 * A ripple at three times the lobe count runs round the whole ring so the edge reads
 * as a wave rather than a rounded triangle: 15% of the lobe height at rest, up to 25%
 * as the top third of the bands fills in, half that between the lobes, drifting
 * slowly against them. It may dip the edge just inside the circle between lobes.
 */
export const BLOB_LOBES = 3;
export const BLOB_RIPPLE_HARMONIC = 3 * BLOB_LOBES;
export const BLOB_RIPPLE_MIN = 0.15;
export const BLOB_RIPPLE_MAX = 0.25;

/**
 * Fills `out` (one entry per ring point, the last closing the loop) with the edge's
 * offset in band units: 0 is the plain circle, 1 a full band. `phase` turns it and
 * `drift` slides the ripple along it.
 */
export function blobEdge(
  bands: Float32Array,
  phase: number,
  out: Float32Array,
  drift = 0,
  lobes: number = BLOB_LOBES,
): void {
  const n = bands.length;
  const points = out.length - 1;
  let peak = 0;
  for (let i = 0; i < n; i++) if (bands[i] > peak) peak = bands[i];
  if (peak <= 0 || points <= 0) {
    out.fill(0);
    return;
  }
  const hiFrom = Math.floor((n * 2) / 3);
  let highs = 0;
  for (let i = hiFrom; i < n; i++) highs += bands[i];
  highs /= Math.max(1, n - hiFrom) * peak;
  const ripple = BLOB_RIPPLE_MIN + (BLOB_RIPPLE_MAX - BLOB_RIPPLE_MIN) * Math.min(1, highs);
  const harmonic = lobes * 3;
  let top = -Infinity;
  for (let i = 0; i <= points; i++) {
    const a = (i / points) * Math.PI * 2 + phase;
    const w = a * (lobes / (Math.PI * 2));
    const f = w - Math.floor(w);
    const env = (1 - Math.cos(f * Math.PI * 2)) / 2;
    const x = f * (n - 1);
    const lo = Math.floor(x);
    const hi = Math.min(n - 1, lo + 1);
    const v = bands[lo] + (bands[hi] - bands[lo]) * (x - lo);
    const d =
      (env * (peak + v)) / 2 + ripple * peak * (0.5 + 0.5 * env) * Math.sin(harmonic * a + drift);
    out[i] = d;
    if (d > top) top = d;
  }
  const gain = top > 0 ? peak / top : 0;
  for (let i = 0; i <= points; i++) out[i] *= gain;
}
