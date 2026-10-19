// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { decayBands } from "../src/features/chat/voice/blob-decay.ts";
import { BLOB_LOBES, blobEdge } from "../src/features/chat/voice/blob-shape.ts";

const POINTS = 96;
const BANDS = 32;
const PHASE = -Math.PI / 2;

function bump(center: number, width = 3, height = 0.8): Float32Array {
  return Float32Array.from({ length: BANDS }, (_, i) => height * Math.exp(-((i - center) ** 2) / (2 * width * width)));
}

function edge(bands: Float32Array, phase = PHASE): Float32Array {
  const out = new Float32Array(POINTS + 1);
  blobEdge(bands, phase, out);
  return out;
}

function peaks(out: Float32Array): number {
  let n = 0;
  for (let i = 0; i < POINTS; i++) {
    const prev = out[(i - 1 + POINTS) % POINTS];
    const next = out[(i + 1) % POINTS];
    if (out[i] > prev && out[i] >= next) n++;
  }
  return n;
}

const max = (a: Float32Array) => a.reduce((m, v) => Math.max(m, v), 0);

test("the edge has three lobes for a voice wherever its energy sits", () => {
  assert.equal(BLOB_LOBES, 3);
  for (const bands of [bump(15.5), bump(8), bump(22), bump(2), bump(29), new Float32Array(BANDS).fill(0.5)]) {
    assert.equal(peaks(edge(bands)), 3);
  }
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
