// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { decayBands } from "../src/features/chat/voice/blob-decay.ts";
import {
  BLOB_PEAKS,
  BLOB_RIPPLE_HARMONIC,
  BLOB_RIPPLE_MAX,
  BLOB_RIPPLE_MIN,
  blobEdge,
  peakAngle,
  spectralCentroid,
} from "../src/features/chat/voice/blob-shape.ts";

const POINTS = 96;
const BANDS = 32;
// Ring points: 0 at the right, 24 at the bottom, 48 at the left, 72 at the top.
const BOTTOM = 24;

function bump(center: number, width = 3, height = 0.8): Float32Array {
  return Float32Array.from({ length: BANDS }, (_, i) => height * Math.exp(-((i - center) ** 2) / (2 * width * width)));
}

function edge(bands: Float32Array, centroid = spectralCentroid(bands), drift = 0): Float32Array {
  const out = new Float32Array(POINTS + 1);
  blobEdge(bands, centroid, out, drift);
  return out;
}

const max = (a: Float32Array) => a.reduce((m, v) => Math.max(m, v), 0);

/** Ring points of the tallest crest on each side of the bottom-top axis. */
function crests(out: Float32Array): { right: number; left: number } {
  let right = BOTTOM;
  let left = BOTTOM;
  for (let i = 0; i < POINTS; i++) {
    const onLeft = i > BOTTOM && i < BOTTOM + 48;
    if (onLeft ? out[i] > out[left] : out[i] > out[right]) {
      if (onLeft) left = i;
      else right = i;
    }
  }
  return { right, left };
}

/** Degrees from the bottom of the ring for point i. */
const fromBottom = (i: number) => Math.abs((((i - BOTTOM) * 360) / POINTS + 540) % 360 - 180);

test("two peaks, mirrored left and right", () => {
  assert.equal(BLOB_PEAKS, 2);
  const out = edge(bump(15.5));
  for (let i = 0; i < POINTS; i++) {
    const mirror = (2 * BOTTOM - i + POINTS) % POINTS;
    assert.ok(Math.abs(out[i] - out[mirror]) < 1e-5, `point ${i}`);
  }
  const { right, left } = crests(out);
  assert.ok(Math.abs(fromBottom(right) - fromBottom(left)) < 1e-9);
  assert.ok(out[right] > 0.95 * max(out) && out[left] > 0.95 * max(out));
  // The axis between them sits well below the crests.
  assert.ok(out[BOTTOM] < 0.5 * max(out) && out[BOTTOM + 48] < 0.5 * max(out));
});

test("the peaks climb the sides as the pitch rises", () => {
  let last = -1;
  for (const centre of [6, 12, 18, 24]) {
    const bands = bump(centre);
    const at = fromBottom(crests(edge(bands)).right);
    assert.ok(at > last, `${centre}: ${at} after ${last}`);
    assert.ok(Math.abs(at - (peakAngle(spectralCentroid(bands)) * 180) / Math.PI) <= 360 / POINTS + 4);
    last = at;
  }
});

test("nothing turns on its own: only the centroid moves the peaks", () => {
  const bands = bump(14);
  const c = spectralCentroid(bands);
  // The ripple's drift is the only time-varying input, and it never moves a crest far.
  for (const drift of [0, 1, 2, 3, 4, 5, 6]) {
    const { right } = crests(edge(bands, c, drift));
    assert.ok(Math.abs(fromBottom(right) - (peakAngle(c) * 180) / Math.PI) < 25, `drift ${drift}`);
  }
});

test("the furthest point reaches the loudest band", () => {
  for (const bands of [bump(15.5), bump(4), bump(26, 2, 0.4), new Float32Array(BANDS).fill(0.3)]) {
    assert.ok(Math.abs(max(edge(bands)) - max(bands)) < 1e-6);
  }
});

test("a gentle ripple rides on the peaks, deeper with bright highs", () => {
  assert.equal(BLOB_RIPPLE_HARMONIC, 6);
  assert.ok(BLOB_RIPPLE_MIN >= 0.15 && BLOB_RIPPLE_MAX <= 0.25);
  const depth = (bands: Float32Array) => {
    let worst = 0;
    for (const drift of [0, 0.7, 1.4, 2.1]) {
      const a = edge(bands, 0.4, drift);
      const b = edge(bands, 0.4, drift + Math.PI);
      for (let i = 0; i <= POINTS; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
    }
    return worst / max(bands);
  };
  const low = depth(bump(4));
  const bright = depth(Float32Array.from(bump(4), (v, i) => Math.max(v, i > 21 ? 0.7 : 0)));
  assert.ok(low > 0.1 && low < 0.6, `low ${low}`);
  assert.ok(bright > low);
});

test("silence is the plain circle", () => {
  assert.equal(max(edge(new Float32Array(BANDS))), 0);
  assert.equal(spectralCentroid(new Float32Array(BANDS)), 0.5);
});

test("the release shrinks the edge in place, so the peaks stay put as it goes round", () => {
  const bands = bump(10);
  const c = spectralCentroid(bands);
  const before = edge(bands, c);
  decayBands(bands, 0, 1000 / 30);
  const after = edge(bands, c);
  const ratio = max(after) / max(before);
  assert.ok(ratio > 0 && ratio < 1);
  for (let i = 0; i <= POINTS; i++) assert.ok(Math.abs(after[i] - before[i] * ratio) < 1e-5);
  assert.ok(Math.abs(spectralCentroid(bands) - c) < 1e-6);
});
