// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { computePeaks, formatSeconds, WAVEFORM_BARS } = await import(
  "../src/features/audio/components/waveform-peaks.ts"
);

test("peaks use the bar count and stay within 0..1", () => {
  const samples = new Float32Array(10_000);
  for (let i = 0; i < samples.length; i += 1)
    samples[i] = Math.sin(i / 20) * (i / samples.length) * 3;
  const peaks = computePeaks([samples], 50);
  assert.equal(peaks.length, 50);
  assert.ok(peaks.every((peak) => peak >= 0 && peak <= 1));
  assert.equal(Math.max(...peaks), 1);
  assert.ok(peaks[49] > peaks[5]);
  assert.equal(computePeaks([samples]).length, WAVEFORM_BARS);
});

test("silence draws flat bars, not amplified noise", () => {
  assert.deepEqual(
    computePeaks([new Float32Array(4000)], 8),
    new Array(8).fill(0),
  );
  const hiss = new Float32Array(4000).fill(0.0001);
  assert.deepEqual(computePeaks([hiss], 8), new Array(8).fill(0));
  assert.deepEqual(computePeaks([], 4), [0, 0, 0, 0]);
});

test("every channel counts, and a clip shorter than the bar count still fills it", () => {
  const left = new Float32Array([0, 0, 0, 0]);
  const right = new Float32Array([0, 0.5, 0, 1]);
  assert.deepEqual(computePeaks([left, right], 4), [0, 0.5, 0, 1]);
  assert.equal(computePeaks([new Float32Array([0.5])], 6).length, 6);
});

test("durations read as m:ss", () => {
  assert.equal(formatSeconds(4.7), "0:05");
  assert.equal(formatSeconds(75), "1:15");
  assert.equal(formatSeconds(3725), "1:02:05");
  assert.equal(formatSeconds(null), "0:00");
});

test("the waveform is neutral, keyboard-playable and static", () => {
  const source = readSrc("features/audio/components/waveform.tsx");
  assert.match(source, /"text-foreground"/);
  assert.match(source, /"text-muted-foreground"/);
  assert.match(source, /event\.key === " "/);
  assert.match(source, /aria-valuetext/);
  assert.doesNotMatch(source, /animate-|transition-/);
});
