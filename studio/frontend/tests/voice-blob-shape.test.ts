// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { decayBands } from "../src/features/chat/voice/blob-decay.ts";
import {
  BLOB_LOBES,
  BLOB_RIPPLE_HARMONIC,
  BLOB_RIPPLE_MAX,
  BLOB_RIPPLE_MIN,
  blobEdge,
} from "../src/features/chat/voice/blob-shape.ts";

const POINTS = 96;
const BANDS = 32;
const PHASE = -Math.PI / 2;

function bump(center: number, width = 3, height = 0.8): Float32Array {
  return Float32Array.from({ length: BANDS }, (_, i) => height * Math.exp(-((i - center) ** 2) / (2 * width * width)));
}

function edge(bands: Float32Array, phase = PHASE, drift = 0): Float32Array {
  const out = new Float32Array(POINTS + 1);
  blobEdge(bands, phase, out, drift);
  return out;
}

function peakValues(out: Float32Array): number[] {
  const found: number[] = [];
  for (let i = 0; i < POINTS; i++) {
    const prev = out[(i - 1 + POINTS) % POINTS];
    const next = out[(i + 1) % POINTS];
    if (out[i] > prev && out[i] >= next) found.push(out[i]);
  }
  return found.sort((a, b) => b - a);
}

const max = (a: Float32Array) => a.reduce((m, v) => Math.max(m, v), 0);

const voices = [bump(15.5), bump(8), bump(22), bump(2), bump(29), new Float32Array(BANDS).fill(0.5)];

test("the edge has three main lobes for a voice wherever its energy sits", () => {
  assert.equal(BLOB_LOBES, 3);
  // The ring between lobes, at this phase: points 24, 56 and 88 of 96.
  const valleys = [24, 56, 88];
  for (const bands of voices) {
    for (let drift = 0; drift < Math.PI * 2; drift += 0.25) {
      const out = edge(bands, PHASE, drift);
      const top = max(out);
      for (const v of valleys) assert.ok(Math.abs(out[v]) < 0.2 * top, `valley ${v}: ${out[v]}`);
      for (let k = 0; k < 3; k++) {
        const lobe = out.slice(valleys[k], k < 2 ? valleys[k + 1] : POINTS);
        const wrap = k < 2 ? lobe : Float32Array.from([...lobe, ...out.slice(0, valleys[0])]);
        assert.ok(max(wrap) > 0.95 * top, `lobe ${k} at drift ${drift}`);
      }
    }
  }
});

test("a ripple rides on the lobes, so the edge undulates and drifts", () => {
  assert.equal(BLOB_RIPPLE_HARMONIC, 9);
  // As it drifts the ripple crests show beside the lobes' own.
  for (const bands of voices) {
    const drifts = Array.from({ length: 25 }, (_, i) => i * 0.25);
    assert.ok(drifts.some((d) => peakValues(edge(bands, PHASE, d)).length > 3));
  }
  const a = edge(bump(12));
  const b = edge(bump(12), PHASE, 1);
  assert.ok(a.some((v, i) => Math.abs(v - b[i]) > 0.01), "drift moves the ripple");
});

test("the ripple is gentle, and the high bands deepen it", () => {
  assert.ok(BLOB_RIPPLE_MIN >= 0.15 && BLOB_RIPPLE_MAX <= 0.25);
  // How far the edge strays from the lobes alone, relative to the lobe height.
  const depth = (bands: Float32Array) => {
    let worst = 0;
    for (const drift of [0, 0.7, 1.4, 2.1]) {
      const rippled = edge(bands, PHASE, drift);
      const smooth = edge(bands, PHASE, drift + Math.PI);
      for (let i = 0; i <= POINTS; i++) worst = Math.max(worst, Math.abs(rippled[i] - smooth[i]));
    }
    return worst / max(bands);
  };
  const low = depth(bump(4));
  const bright = depth(Float32Array.from(bump(4), (v, i) => Math.max(v, i > 21 ? 0.7 : 0)));
  assert.ok(low > 0.1 && low < 0.6, `low ${low}`);
  assert.ok(bright > low, `bright ${bright} vs ${low}`);
});

test("the lobes are evenly spaced and the loop closes", () => {
  const out = edge(bump(12));
  const third = POINTS / 3;
  for (let i = 0; i < POINTS; i++) {
    assert.ok(Math.abs(out[i] - out[(i + third) % POINTS]) < 1e-5, `point ${i}`);
  }
  assert.ok(Math.abs(out[0] - out[POINTS]) < 1e-5);
});

test("the furthest point reaches the loudest band, as the two-hump edge did", () => {
  for (const bands of [bump(15.5), bump(4), bump(26, 2, 0.4), new Float32Array(BANDS).fill(0.3)]) {
    assert.ok(Math.abs(max(edge(bands)) - max(bands)) < 1e-6);
  }
});

test("silence is the plain circle", () => {
  assert.equal(max(edge(new Float32Array(BANDS))), 0);
});

test("the release shrinks the edge in proportion, so it slides back to round", () => {
  const bands = bump(10);
  const before = edge(bands);
  decayBands(bands, 0, 1000 / 30);
  const after = edge(bands);
  const ratio = max(after) / max(before);
  assert.ok(ratio > 0 && ratio < 1);
  for (let i = 0; i <= POINTS; i++) assert.ok(Math.abs(after[i] - before[i] * ratio) < 1e-5);
});
