// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  BLOB_DECAY_TAU_MS,
  BLOB_FLAT,
  decayBands,
  decayFactor,
  easeToward,
  isFlat,
} from "../src/features/chat/voice/blob-decay.ts";

const FRAME = 1000 / 30;

function framesToFlat(start: number): number {
  const bands = new Float32Array(32).fill(start);
  let level = start;
  let frames = 0;
  while (!isFlat(bands, level)) {
    level = decayBands(bands, level, FRAME);
    frames++;
    assert.ok(frames < 1000);
  }
  return frames;
}

test("a full wave settles flat in roughly a third of a second, easing out", () => {
  const ms = framesToFlat(1) * FRAME;
  assert.ok(ms >= 250 && ms <= 400, `${ms} ms`);
  // Ease-out: the first frame moves the edge far more than the last.
  const first = 1 - decayFactor(FRAME);
  const atEnd = BLOB_FLAT * 2 * (1 - decayFactor(FRAME));
  assert.ok(first > 10 * atEnd);
});

test("decay never overshoots below zero or grows", () => {
  const bands = Float32Array.from([1, 0.5, 0, 0.2]);
  let level = 0.8;
  for (let i = 0; i < 40; i++) {
    const before = Float32Array.from(bands);
    level = decayBands(bands, level, FRAME);
    bands.forEach((v, j) => assert.ok(v >= 0 && v <= before[j]));
  }
  assert.ok(level >= 0 && isFlat(bands, level));
});

test("decay does not depend on the frame rate", () => {
  const a = new Float32Array([1]);
  const b = new Float32Array([1]);
  decayBands(a, 1, 100);
  for (let i = 0; i < 10; i++) decayBands(b, 1, 10);
  assert.ok(Math.abs(a[0] - b[0]) < 1e-6);
  assert.equal(decayFactor(0), 1);
  assert.equal(decayFactor(-5), 1);
  assert.ok(Math.abs(decayFactor(BLOB_DECAY_TAU_MS) - Math.exp(-1)) < 1e-12);
});

test("a starting voice eases in instead of popping, and rises faster than it falls", () => {
  const rise = easeToward(0, 1, FRAME);
  assert.ok(rise > 0.3 && rise < 0.6, `${rise}`);
  const fall = 1 - easeToward(1, 0, FRAME);
  assert.ok(fall < rise);
  assert.equal(easeToward(0.4, 0.4, FRAME), 0.4);
});
