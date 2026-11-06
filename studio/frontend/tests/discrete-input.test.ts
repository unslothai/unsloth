// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DISCRETE_INPUT_EVENTS,
  DISCRETE_INPUT_WINDOW_MS,
  createDiscreteInputTracker,
  discreteInputPending,
} from "../src/components/assistant-ui/discrete-input.ts";

import { readSrc } from "./helpers/kit.ts";

test("no input ever seen is never pending", () => {
  assert.equal(discreteInputPending(Number.NEGATIVE_INFINITY, 0), false);
  assert.equal(discreteInputPending(Number.NEGATIVE_INFINITY, 1e9), false);
});

test("the window is half-open: pending strictly inside, not at the edge", () => {
  assert.equal(DISCRETE_INPUT_WINDOW_MS, 50);
  assert.equal(discreteInputPending(100, 100), true);
  assert.equal(discreteInputPending(100, 149.9), true);
  assert.equal(discreteInputPending(100, 150), false);
  assert.equal(discreteInputPending(100, 120, 10), false);
});

test("the tracker counts every discrete event through the window", () => {
  assert.deepEqual([...DISCRETE_INPUT_EVENTS], ["keydown", "pointerdown", "touchstart"]);
  for (const type of DISCRETE_INPUT_EVENTS) {
    let now = 1000;
    const target = new EventTarget();
    const tracker = createDiscreteInputTracker(target, () => now);
    assert.equal(tracker.pending(), false);
    target.dispatchEvent(new Event(type));
    now = 1020;
    assert.equal(tracker.pending(), true, type);
    now = 1049;
    assert.equal(tracker.pending(), true, type);
    now = 1050;
    assert.equal(tracker.pending(), false, type);
    tracker.dispose();
  }
});

test("continuous input never counts, matching Chromium's isInputPending default", () => {
  let now = 1000;
  const target = new EventTarget();
  const tracker = createDiscreteInputTracker(target, () => now);
  for (const type of ["wheel", "pointermove", "mousemove", "touchmove", "scroll", "keyup", "pointerup"]) {
    target.dispatchEvent(new Event(type));
    now += 1;
    assert.equal(tracker.pending(), false, type);
  }
  tracker.dispose();
});

test("dispose stops recording", () => {
  let now = 0;
  const target = new EventTarget();
  const tracker = createDiscreteInputTracker(target, () => now);
  tracker.dispose();
  target.dispatchEvent(new Event("keydown"));
  now = 10;
  assert.equal(tracker.pending(), false);
});

test("the stream loop prefers the native signal and only then the tracker", () => {
  const source = readSrc("components/assistant-ui/markdown-text.tsx");
  assert.match(
    source,
    /if \(scheduling\?\.isInputPending\) \{\s*return scheduling\.isInputPending\(\);\s*\}/,
  );
  assert.ok(source.includes("discreteInput ??= createDiscreteInputTracker(document);"));
  assert.ok(!source.includes("includeContinuous: true"));
});
