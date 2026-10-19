// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The orb's spectrum edge: the log bands folded once round the ring, low to high and
 * back, mirrored and slowly turning, so the voice's energy makes a big blob on one side
 * and a little one opposite. Drawn from 32 control points, joined by one
 * closed path of cubic beziers whose tangents are continuous everywhere. Only the
 * control points change per frame.
 */
/**
 * A full band's reach past the ball's 70 px radius at the orb's full size. 40% over the
 * 16 px it was, so the peaks read as waves rather than side bumps.
 */
export const BLOB_AMP_PX = 22.4;

/**
 * How far the quietest stretch of the edge sinks inside the ball, as a share of the
 * loudest band, so the gaps between blobs pinch in and the shape reads as stretched.
 */
export const BLOB_PINCH = 0.4;

/**
 * Control points evenly round the ring, 11.25 degrees apart: fewer and the blobs lose
 * their shape, more only lengthens the clip path the browser re-parses every frame.
 */
export const BLOB_CONTROLS = 32;
const STEP = (2 * Math.PI) / BLOB_CONTROLS;

export interface BlobControls {
  /** Screen angle of each point (0 at the right, clockwise), increasing round the ring. */
  angle: Float32Array;
  /** Each point's offset past the circle, in band units (1 is a full band). */
  offset: Float32Array;
}

export function createBlobControls(): BlobControls {
  return {
    angle: new Float32Array(BLOB_CONTROLS),
    offset: new Float32Array(BLOB_CONTROLS),
  };
}

/**
 * The original edge: the log bands folded once round the ring from low to high and back,
 * so it has one mirror axis, turned by `phase` (screen radians). Wherever the voice's
 * energy sits makes the big blob and the opposite end the little one.
 */
export function foldControls(bands: Float32Array, phase: number, ctl: BlobControls): void {
  const n = bands.length;
  let peak = 0;
  for (let i = 0; i < n; i++) if (bands[i] > peak) peak = bands[i];
  for (let i = 0; i < BLOB_CONTROLS; i++) {
    // The points turn with the edge, so each keeps its band and nothing shimmers.
    const a = i * STEP - phase;
    const x = ((1 - Math.cos(i * STEP)) / 2) * (n - 1);
    const lo = Math.floor(x);
    const hi = Math.min(n - 1, lo + 1);
    let v = bands[lo] + (bands[hi] - bands[lo]) * (x - lo);
    // In proportion to the loudest band, so the release still shrinks it evenly.
    if (peak > 0) v -= BLOB_PINCH * peak * (1 - v / peak) ** 2;
    ctl.angle[i] = a;
    ctl.offset[i] = v;
  }
}

const slopes = new Float32Array(BLOB_CONTROLS);

/**
 * One closed path through the control points, at radius `base + offset * amp` around
 * (`cx`, `cy`), all scaled by `k`. The radius is a monotone cubic (PCHIP) in the angle, so
 * it never overshoots a control point and its slope is zero at every peak, valley and
 * seam; each span becomes one cubic bezier, so tangents are continuous everywhere.
 */
export function blobBezierPath(ctl: BlobControls, base: number, amp: number, cx: number, cy: number, k = 1): string {
  const n = BLOB_CONTROLS;
  const { angle, offset } = ctl;
  const span = (i: number) => (i === n - 1 ? angle[0] + 2 * Math.PI : angle[i + 1]) - angle[i];
  const rise = (i: number) => (offset[(i + 1) % n] - offset[i]) * amp;
  for (let i = 0; i < n; i++) {
    const p = (i - 1 + n) % n;
    const h0 = span(p);
    const h1 = span(i);
    const d0 = rise(p) / h0;
    const d1 = rise(i) / h1;
    if (d0 * d1 <= 0) slopes[i] = 0;
    else {
      const w0 = 2 * h1 + h0;
      const w1 = h1 + 2 * h0;
      slopes[i] = (w0 + w1) / (w0 / d0 + w1 / d1);
    }
  }
  let d = "";
  for (let i = 0; i <= n; i++) {
    const j = i % n;
    const a = angle[j];
    const r = base + offset[j] * amp;
    const x = cx + Math.cos(a) * r * k;
    const y = cy + Math.sin(a) * r * k;
    if (i === 0) {
      d = `M${x.toFixed(1)} ${y.toFixed(1)}`;
      continue;
    }
    const q = i - 1;
    const aq = angle[q];
    const rq = base + offset[q] * amp;
    const h = span(q);
    // The circle-true handle length for a span of h, applied to the curve's derivative.
    const l = (4 / 3) * Math.tan(h / 4) * k;
    const c1x = cx + Math.cos(aq) * rq * k + (slopes[q] * Math.cos(aq) - rq * Math.sin(aq)) * l;
    const c1y = cy + Math.sin(aq) * rq * k + (slopes[q] * Math.sin(aq) + rq * Math.cos(aq)) * l;
    const c2x = x - (slopes[j] * Math.cos(a) - r * Math.sin(a)) * l;
    const c2y = y - (slopes[j] * Math.sin(a) + r * Math.cos(a)) * l;
    d += `C${c1x.toFixed(1)} ${c1y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return `${d}Z`;
}
