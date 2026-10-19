// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The orb's spectrum edge as three lobes. Each lobe sweeps the bands low to high
 * under a cosine envelope that is zero between lobes, so the seam where one lobe's
 * high bands meet the next one's low bands never shows. Half of each lobe is the
 * loudest band, so a voice whose energy sits off-centre still reads as three lobes,
 * and the result is scaled so the furthest point is exactly the loudest band, the
 * same reach the mirrored two-hump edge had.
 */
export const BLOB_LOBES = 3;

/**
 * Fills `out` (one entry per ring point, the last closing the loop) with the edge's
 * offset in band units: 0 is the plain circle, 1 a full band. `phase` turns it.
 */
export function blobEdge(
  bands: Float32Array,
  phase: number,
  out: Float32Array,
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
  let top = 0;
  for (let i = 0; i <= points; i++) {
    const w = ((i / points) * Math.PI * 2 + phase) * (lobes / (Math.PI * 2));
    const f = w - Math.floor(w);
    const env = (1 - Math.cos(f * Math.PI * 2)) / 2;
    const x = f * (n - 1);
    const lo = Math.floor(x);
    const hi = Math.min(n - 1, lo + 1);
    const v = bands[lo] + (bands[hi] - bands[lo]) * (x - lo);
    const d = (env * (peak + v)) / 2;
    out[i] = d;
    if (d > top) top = d;
  }
  const gain = top > 0 ? peak / top : 0;
  for (let i = 0; i <= points; i++) out[i] *= gain;
}
