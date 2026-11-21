// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  SpeechSegmenter,
  type SpeechSegment,
} from "../src/features/chat/adapters/dictation-segmenter.ts";

const RATE = 16_000;

function frame(ms: number, amplitude: number): Float32Array {
  return new Float32Array(Math.round((ms / 1000) * RATE)).fill(amplitude);
}

function feed(
  segmenter: SpeechSegmenter,
  frames: Float32Array[],
): SpeechSegment[] {
  const segments: SpeechSegment[] = [];
  for (const sample of frames) {
    const segment = segmenter.push(sample);
    if (segment) segments.push(segment);
  }
  return segments;
}

test("a pause after the minimum duration cuts a sentence", () => {
  const segmenter = new SpeechSegmenter({
    sampleRate: RATE,
    preRollMs: 200,
    minSegmentMs: 500,
    maxSegmentMs: 5000,
    silenceCutMs: 300,
    trimMarginMs: 0,
    minSpeechMs: 50,
  });
  const segments = feed(segmenter, [
    frame(500, 0),
    frame(600, 0.1),
    frame(500, 0),
  ]);
  assert.equal(segments.length, 1);
  const [segment] = segments;
  assert.equal(segment.sampleRate, RATE);
  // The pause is trimmed away, leaving the spoken part.
  assert.ok(segment.durationMs >= 550 && segment.durationMs <= 700, `${segment.durationMs}`);
});

test("the pre-roll keeps a quiet first syllable the detector missed", () => {
  const segmenter = new SpeechSegmenter({
    sampleRate: RATE,
    preRollMs: 200,
    minSegmentMs: 300,
    maxSegmentMs: 5000,
    silenceCutMs: 300,
    trimMarginMs: 0,
    minSpeechMs: 50,
  });
  const quiet = frame(40, 0.008);
  const segments = feed(segmenter, [
    frame(200, 0),
    quiet,
    frame(400, 0.1),
    frame(500, 0),
  ]);
  assert.equal(segments.length, 1);
  // 0.008 is below the speech threshold but above the trim floor, so the first
  // sample survives in front of the loud part.
  assert.ok(Math.abs(segments[0].samples[0]) >= 0.004);
});

test("continuous speech is cut at the hard maximum", () => {
  const segmenter = new SpeechSegmenter({
    sampleRate: RATE,
    preRollMs: 0,
    minSegmentMs: 500,
    maxSegmentMs: 1000,
    silenceCutMs: 300,
    trimMarginMs: 0,
    minSpeechMs: 50,
  });
  const frames: Float32Array[] = [];
  for (let index = 0; index < 60; index += 1) frames.push(frame(20, 0.1));
  const segments = feed(segmenter, frames);
  assert.ok(segments.length >= 1);
  assert.ok(segments[0].durationMs >= 950 && segments[0].durationMs <= 1050);
});

test("flush returns the sentence still in progress exactly once", () => {
  const segmenter = new SpeechSegmenter({
    sampleRate: RATE,
    preRollMs: 100,
    minSegmentMs: 5000,
    maxSegmentMs: 14_000,
    silenceCutMs: 300,
    trimMarginMs: 0,
    minSpeechMs: 50,
  });
  assert.deepEqual(feed(segmenter, [frame(600, 0.1)]), []);
  const segment = segmenter.flush();
  assert.ok(segment);
  assert.ok(segment.durationMs >= 550 && segment.durationMs <= 650);
  assert.equal(segmenter.flush(), null);
});

test("silence never produces a segment", () => {
  const segmenter = new SpeechSegmenter({
    sampleRate: RATE,
    preRollMs: 200,
    minSegmentMs: 500,
    maxSegmentMs: 5000,
    silenceCutMs: 300,
  });
  assert.deepEqual(feed(segmenter, [frame(2000, 0)]), []);
  assert.equal(segmenter.flush(), null);
});
