// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { decayBands } from "../src/features/chat/voice/blob-decay.ts";
import {
  BLOB_AMP_PX,
  BLOB_CONTROLS,
  BLOB_PINCH,
  blobBezierPath,
  createBlobControls,
  foldControls,
} from "../src/features/chat/voice/blob-shape.ts";

const BANDS = 32;
const BASE = 73;
const REST = -Math.PI / 2;
type Pt = [number, number];
type Seg = { p0: Pt; c1: Pt; c2: Pt; p1: Pt };

function bump(center: number, width = 3, height = 0.8): Float32Array {
  return Float32Array.from({ length: BANDS }, (_, i) => height * Math.exp(-((i - center) ** 2) / (2 * width * width)));
}

function shape(bands: Float32Array, phase = REST) {
  const ctl = createBlobControls();
  foldControls(bands, phase, ctl);
  const d = blobBezierPath(ctl, BASE, BLOB_AMP_PX, 0, 0);
  const nums = (d.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);
  const segs: Seg[] = [];
  let at: Pt = [nums[0], nums[1]];
  for (let i = 2; i + 5 < nums.length; i += 6) {
    const seg = {
      p0: at,
      c1: [nums[i], nums[i + 1]] as Pt,
      c2: [nums[i + 2], nums[i + 3]] as Pt,
      p1: [nums[i + 4], nums[i + 5]] as Pt,
    };
    segs.push(seg);
    at = seg.p1;
  }
  return { ctl, d, segs };
}

function sample(segs: Seg[], per = 40): Pt[] {
  const out: Pt[] = [];
  for (const { p0, c1, c2, p1 } of segs) {
    for (let s = 0; s < per; s++) {
      const t = s / per;
      const u = 1 - t;
      const b = (a: number, b1: number, c: number, e: number) =>
        u * u * u * a + 3 * u * u * t * b1 + 3 * u * t * t * c + t * t * t * e;
      out.push([b(p0[0], c1[0], c2[0], p1[0]), b(p0[1], c1[1], c2[1], p1[1])]);
    }
  }
  return out;
}

const radius = ([x, y]: Pt) => Math.hypot(x, y);
const max = (a: ArrayLike<number>) => Array.from(a).reduce((m, v) => Math.max(m, v), 0);

/** The original edge's offset at screen angle `a` (e0f704eba0^ voice-orb.tsx), plus the pinch. */
function original(bands: Float32Array, a: number, phase: number): number {
  const x = ((1 - Math.cos(a + phase)) / 2) * (BANDS - 1);
  const lo = Math.floor(x);
  const hi = Math.min(BANDS - 1, lo + 1);
  const v = bands[lo] + (bands[hi] - bands[lo]) * (x - lo);
  const peak = max(bands);
  return v - BLOB_PINCH * peak * (1 - v / peak) ** 2;
}

test("one closed path of 32 cubic beziers, short enough to re-parse every frame", () => {
  assert.equal(BLOB_CONTROLS, 32);
  assert.ok(shape(bump(14)).d.length < 1200);
  const { d, segs } = shape(bump(14));
  assert.equal(segs.length, BLOB_CONTROLS);
  assert.match(d, new RegExp(`^M[^C]+(C[^C]+){${BLOB_CONTROLS}}Z$`));
});

test("every control point is the original edge with its gaps pinched in, at any turn", () => {
  for (const phase of [REST, 0.7, 2.9]) {
    const bands = bump(9);
    const { ctl } = shape(bands, phase);
    for (let i = 0; i < BLOB_CONTROLS; i++) {
      // The points turn with the edge, so each keeps its band.
      assert.ok(Math.abs(ctl.angle[i] - ((i * 2 * Math.PI) / BLOB_CONTROLS - phase)) < 1e-6);
      assert.ok(Math.abs(ctl.offset[i] - original(bands, ctl.angle[i], phase)) < 1e-6);
    }
  }
});

test("tangents are continuous at every joint", () => {
  for (const phase of [REST, 1.3]) {
    const { segs } = shape(bump(12), phase);
    for (let i = 0; i < segs.length; i++) {
      const into = segs[i];
      const out = segs[(i + 1) % segs.length];
      const a: Pt = [into.p1[0] - into.c2[0], into.p1[1] - into.c2[1]];
      const b: Pt = [out.c1[0] - out.p0[0], out.c1[1] - out.p0[1]];
      const cos = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b));
      // Coordinates are rounded to 0.1 px in the path, hence the small tolerance.
      assert.ok(cos > 0.995, `joint ${i} at phase ${phase}: ${cos}`);
    }
  }
});

test("at rest, low bands at the bottom and high at the top, mirrored left and right", () => {
  const n = BLOB_CONTROLS;
  // At rest point 0 sits at the bottom (screen angle pi/2) and point n/2 at the top.
  assert.ok(Math.abs(shape(bump(0)).ctl.angle[0] - Math.PI / 2) < 1e-6);
  const bottom = 0;
  const top = n / 2;
  const low = shape(bump(0)).ctl.offset;
  const high = shape(bump(29)).ctl.offset;
  assert.equal(low[bottom], max(low));
  assert.ok(low[top] < 0.1 * low[bottom]);
  assert.ok(high[top] > 10 * high[bottom]);
  for (let j = 1; j < n / 2; j++) assert.ok(Math.abs(low[j] - low[n - j]) < 1e-5);
});

test("a voice reads as a big blob on one side and a little one on the other", () => {
  const bands = Float32Array.from(
    { length: BANDS },
    (_, i) => 0.9 * Math.exp(-((i - 6) ** 2) / 18) + 0.3 * Math.exp(-((i - 26) ** 2) / 8),
  );
  const { ctl } = shape(bands);
  const n = BLOB_CONTROLS;
  // Lower half of the screen (sin > 0) against the upper half.
  const lower = Array.from(ctl.offset).filter((_, i) => Math.sin(ctl.angle[i]) > 0);
  const upper = Array.from(ctl.offset).filter((_, i) => Math.sin(ctl.angle[i]) < 0);
  const big = max(lower);
  const little = max(upper);
  assert.ok(big > 2 * little && little > 0.1, `${big} vs ${little}`);
});

test("the curve never overshoots: its furthest point is the loudest band at +40% reach", () => {
  assert.ok(Math.abs(BLOB_AMP_PX - 16 * 1.4) < 1e-9);
  for (const bands of [bump(15.5), new Float32Array(BANDS).fill(1)]) {
    const pts = sample(shape(bands).segs);
    const far = max(pts.map(radius));
    assert.ok(far <= BASE + max(bands) * BLOB_AMP_PX + 0.3, `${far}`);
    // The gaps pinch in, never by more than the pinch.
    assert.ok(Math.min(...pts.map(radius)) >= BASE - BLOB_PINCH * max(bands) * BLOB_AMP_PX - 0.3);
  }
});

test("the gaps between blobs pinch inside the ball by 6-10 px on a loud voice", () => {
  // The loudest band on a control point (band 0 sits on point 0), so its reach is exact.
  const pts = sample(shape(bump(0, 3, 1)).segs);
  const dip = BASE - Math.min(...pts.map(radius));
  assert.ok(dip >= 6 && dip <= 10, `${dip}`);
  // And only where it is quiet: the loud blob is untouched.
  assert.ok(Math.abs(max(pts.map(radius)) - (BASE + BLOB_AMP_PX)) < 0.3);
});

test("silence is the plain circle", () => {
  const pts = sample(shape(new Float32Array(BANDS)).segs);
  for (const p of pts) assert.ok(Math.abs(radius(p) - BASE) < 0.3);
});

test("the release shrinks every control point in proportion, so the peaks stay put", () => {
  const bands = bump(10);
  const before = shape(bands, 0.35).ctl;
  const offsets = Float32Array.from(before.offset);
  decayBands(bands, 0, 1000 / 30);
  const after = shape(bands, 0.35).ctl;
  const ratio = max(after.offset) / max(offsets);
  assert.ok(ratio > 0 && ratio < 1);
  for (let i = 0; i < BLOB_CONTROLS; i++) {
    assert.ok(Math.abs(after.offset[i] - offsets[i] * ratio) < 1e-5);
    assert.equal(after.angle[i], before.angle[i]);
  }
});
