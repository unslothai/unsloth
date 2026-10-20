// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SPECTRUM_BANDS,
  bandEdges,
  createBandState,
  shapeBands,
} from "../src/features/chat/voice/voice-spectrum.ts";

const SR = 48000;
const FFT = 2048;
const BIN = SR / FFT;

/** A voiced frame: harmonics of f0 falling 12 dB an octave, two formants, in dB per bin. */
function voice(f0: number, f1: number, f2: number, gain = 1): Float32Array {
  const raw = new Float32Array(FFT / 2);
  for (let i = 0; i < raw.length; i++) {
    const f = i * BIN;
    let p = 1e-10;
    for (let k = 1; k * f0 < 5000; k++) {
      const h = k * f0;
      const form =
        8 * Math.exp(-(((h - f1) / 180) ** 2)) +
        3 * Math.exp(-(((h - f2) / 260) ** 2)) +
        1;
      const a = (gain * form) / k ** 2;
      p += a * a * Math.exp(-(((f - h) / BIN) ** 2));
    }
    raw[i] = Math.max(-95, 10 * Math.log10(p) - 40);
  }
  return raw;
}

/** Talks for `ms` cycling vowels, returns the last frame's bands. */
function talk(f0: number, ms = 3000): Float32Array {
  const vowels = [
    [700, 1200],
    [400, 2000],
    [550, 1700],
    [300, 2300],
    [650, 1000],
  ];
  const frames = vowels.map(([a, b]) => voice(f0, a, b));
  const state = createBandState(new Float32Array(FFT / 2), bandEdges(SR, FFT));
  const out = new Float32Array(SPECTRUM_BANDS);
  for (let t = 0; t <= ms; t += 16) {
    state.raw.set(frames[Math.floor(t / 250) % frames.length]);
    shapeBands(state, out, t);
  }
  return out;
}

const mean = (a: ArrayLike<number>) =>
  Array.from(a).reduce((s, v) => s + v, 0) / a.length;

test("the bands cover the speaking range, each band its own bins", () => {
  const edges = bandEdges(SR, FFT);
  assert.ok(
    edges[0] * BIN < 100 &&
      edges[SPECTRUM_BANDS] * BIN > 3800 &&
      edges[SPECTRUM_BANDS] * BIN < 4200,
  );
});

test("a low, a mid and a high voice all light both halves of the band range", () => {
  for (const f0 of [110, 170, 250]) {
    const out = talk(f0);
    const lowHalf = mean(out.slice(0, SPECTRUM_BANDS / 2));
    const highHalf = mean(out.slice(SPECTRUM_BANDS / 2));
    const ratio = Math.min(lowHalf, highHalf) / Math.max(lowHalf, highHalf);
    assert.ok(
      ratio > 0.5,
      `f0 ${f0}: ${lowHalf.toFixed(2)} vs ${highHalf.toFixed(2)}`,
    );
  }
});

test("the shape still has lobes: it does not swell round as a whole", () => {
  const out = talk(110);
  assert.ok(Math.min(...out) < 0.5 * Math.max(...out));
});

test("the size still follows loudness: a quiet voice draws a smaller edge", () => {
  const quiet = createBandState(new Float32Array(FFT / 2), bandEdges(SR, FFT));
  const out = new Float32Array(SPECTRUM_BANDS);
  const frame = voice(110, 700, 1200, 0.05);
  let level = 0;
  for (let t = 0; t <= 2000; t += 16) {
    quiet.raw.set(frame);
    level = shapeBands(quiet, out, t);
  }
  const loud = talk(110);
  assert.ok(Math.max(...out) < 0.7 * Math.max(...loud));
  assert.ok(level > 0);
});

test("silence draws nothing", () => {
  const state = createBandState(
    new Float32Array(FFT / 2).fill(-140),
    bandEdges(SR, FFT),
  );
  const out = new Float32Array(SPECTRUM_BANDS);
  for (let t = 0; t < 500; t += 16) assert.equal(shapeBands(state, out, t), 0);
  assert.equal(Math.max(...out), 0);
});
