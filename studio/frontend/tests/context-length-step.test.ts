// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  CONTEXT_LENGTH_SLIDER_STEPS,
  getContextLengthSliderBounds,
  snapContextLengthToStep,
} from "../src/features/model-picker/model-config/context-length-step.ts";

test("the picker exposes fine and fixed adjustment modes", () => {
  assert.deepEqual(
    CONTEXT_LENGTH_SLIDER_STEPS.map(({ value }) => value),
    [1, 4096, 8192],
  );
});

test("fine adjustment preserves the natural context bounds", () => {
  assert.deepEqual(getContextLengthSliderBounds(128, 32768, 1), {
    min: 128,
    max: 32768,
  });
  assert.equal(snapContextLengthToStep(32761, 128, 32768, 1), 32761);
});

test("fixed adjustment snaps to clean token multiples", () => {
  assert.deepEqual(getContextLengthSliderBounds(128, 65536, 4096), {
    min: 4096,
    max: 65536,
  });
  assert.equal(snapContextLengthToStep(32761, 128, 65536, 4096), 32768);
  assert.equal(snapContextLengthToStep(3000, 128, 65536, 4096), 4096);
});

test("fixed adjustment keeps the highest representable value below a non-aligned ceiling", () => {
  assert.deepEqual(getContextLengthSliderBounds(128, 32761, 4096), {
    min: 4096,
    max: 28672,
  });
  assert.equal(snapContextLengthToStep(32761, 128, 32761, 4096), 28672);
});

test("a step larger than the available context never exceeds the model limit", () => {
  assert.deepEqual(getContextLengthSliderBounds(128, 4096, 8192), {
    min: 128,
    max: 4096,
  });
  assert.equal(snapContextLengthToStep(3000, 128, 4096, 8192), 3000);
  assert.equal(snapContextLengthToStep(10000, 128, 4096, 8192), 4096);
  assert.equal(snapContextLengthToStep(128, 128, 128, 8192), 128);
});

test("fractional and invalid steps fall back to a finite token value", () => {
  for (const step of [0.5, 0, -1, NaN, Infinity]) {
    assert.equal(snapContextLengthToStep(3000, 128, 4096, step), 3000);
  }
});
