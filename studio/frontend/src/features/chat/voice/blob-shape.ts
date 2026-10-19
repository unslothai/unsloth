// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The orb's spectrum edge: two mirrored peaks placed by pitch. The ring runs from the
 * lowest band at the bottom to the highest at the top on both sides, and the peaks sit
 * where the spectrum's centroid falls on it, so a higher voice lifts them up the sides
 * and a low one draws them together at the bottom. Nothing turns on its own: the angle
 * moves only with the centroid, which the caller eases so the peaks glide.
 *
 * A ripple, three crests up each side, rides on top so the edge reads as a wave: 15% of
 * the peak height at rest, up to 25% as the top third of the bands fills in, half that
 * away from the peaks. The result is scaled so its furthest point is the loudest band.
 */
export const BLOB_PEAKS = 2;
export const BLOB_RIPPLE_HARMONIC = 3 * BLOB_PEAKS;
export const BLOB_RIPPLE_MIN = 0.15;
export const BLOB_RIPPLE_MAX = 0.25;
/** A peak's half-width along the ring, in the 0 (bottom) .. 1 (top) position. */
const PEAK_WIDTH = 0.2;

/** Where the spectrum's energy sits, 0 the lowest band to 1 the highest; 0.5 when silent. */
export function spectralCentroid(bands: Float32Array): number {
  let sum = 0;
  let weighted = 0;
  for (let i = 0; i < bands.length; i++) {
    sum += bands[i];
    weighted += bands[i] * i;
  }
  if (sum <= 0 || bands.length < 2) return 0.5;
  return weighted / sum / (bands.length - 1);
}

/** Angle from the bottom of the ring (radians, either side) where the peaks sit. */
export function peakAngle(centroid: number): number {
  return Math.acos(1 - 2 * Math.min(1, Math.max(0, centroid)));
}

/**
 * Fills `out` (one entry per ring point, the last closing the loop, point 0 at the right
 * and counting clockwise on screen) with the edge's offset in band units: 0 is the plain
 * circle, 1 a full band. `centroid` places the peaks; `drift` slides the ripple.
 */
export function blobEdge(
  bands: Float32Array,
  centroid: number,
  out: Float32Array,
  drift = 0,
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
  const c = Math.min(1, Math.max(0, centroid));
  let top = -Infinity;
  for (let i = 0; i <= points; i++) {
    const a = (i / points) * Math.PI * 2;
    // 0 at the bottom (a = pi/2 on screen), 1 at the top, the same on both sides.
    const pos = (1 - Math.sin(a)) / 2;
    // Measured from the bottom on both sides, so the ripple mirrors as the peaks do.
    const fromBottom = Math.acos(Math.min(1, Math.max(-1, Math.sin(a))));
    const u = (pos - c) / PEAK_WIDTH;
    const lobe = Math.abs(u) < 1 ? (1 + Math.cos(Math.PI * u)) / 2 : 0;
    const x = pos * (n - 1);
    const lo = Math.floor(x);
    const hi = Math.min(n - 1, lo + 1);
    const v = bands[lo] + (bands[hi] - bands[lo]) * (x - lo);
    const d =
      (lobe * (peak + v)) / 2 +
      ripple * peak * (0.5 + 0.5 * lobe) * Math.sin(BLOB_RIPPLE_HARMONIC * fromBottom + drift);
    out[i] = d;
    if (d > top) top = d;
  }
  const gain = top > 0 ? peak / top : 0;
  for (let i = 0; i <= points; i++) out[i] *= gain;
}
