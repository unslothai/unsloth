// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const BLOB_AMP_PX = 22.4;

export const BLOB_PINCH = 0.4;

export const BLOB_CONTROLS = 32;
const STEP = (2 * Math.PI) / BLOB_CONTROLS;

export interface BlobControls {
  angle: Float32Array;
  offset: Float32Array;
}

export function createBlobControls(): BlobControls {
  return {
    angle: new Float32Array(BLOB_CONTROLS),
    offset: new Float32Array(BLOB_CONTROLS),
  };
}

export function foldControls(bands: Float32Array, phase: number, ctl: BlobControls): void {
  const n = bands.length;
  let peak = 0;
  for (let i = 0; i < n; i++) if (bands[i] > peak) peak = bands[i];
  for (let i = 0; i < BLOB_CONTROLS; i++) {
    const a = i * STEP - phase;
    const x = ((1 - Math.cos(i * STEP)) / 2) * (n - 1);
    const lo = Math.floor(x);
    const hi = Math.min(n - 1, lo + 1);
    let v = bands[lo] + (bands[hi] - bands[lo]) * (x - lo);
    if (peak > 0) v -= BLOB_PINCH * peak * (1 - v / peak) ** 2;
    ctl.angle[i] = a;
    ctl.offset[i] = v;
  }
}

const slopes = new Float32Array(BLOB_CONTROLS);

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
    const l = (4 / 3) * Math.tan(h / 4) * k;
    const c1x = cx + Math.cos(aq) * rq * k + (slopes[q] * Math.cos(aq) - rq * Math.sin(aq)) * l;
    const c1y = cy + Math.sin(aq) * rq * k + (slopes[q] * Math.sin(aq) + rq * Math.cos(aq)) * l;
    const c2x = x - (slopes[j] * Math.cos(a) - r * Math.sin(a)) * l;
    const c2y = y - (slopes[j] * Math.sin(a) + r * Math.cos(a)) * l;
    d += `C${c1x.toFixed(1)} ${c1y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return `${d}Z`;
}
