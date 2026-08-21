// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  AUTOSCROLL_EDGE,
  AUTOSCROLL_MAX_STEP,
  autoscrollDelta,
} from "../src/features/chat/prompt-storage/autoscroll.ts";

// A pane 400px tall sitting 100px down the viewport.
const TOP = 100;
const BOTTOM = 500;

test("a pointer in the middle of the pane does not scroll it", () => {
  assert.equal(autoscrollDelta(300, TOP, BOTTOM), 0);
  assert.equal(autoscrollDelta(TOP + AUTOSCROLL_EDGE, TOP, BOTTOM), 0);
  assert.equal(autoscrollDelta(BOTTOM - AUTOSCROLL_EDGE, TOP, BOTTOM), 0);
});

test("nearing the top edge scrolls up, nearing the bottom scrolls down", () => {
  assert.ok(autoscrollDelta(TOP + 10, TOP, BOTTOM) < 0);
  assert.ok(autoscrollDelta(BOTTOM - 10, TOP, BOTTOM) > 0);
});

test("the step ramps with depth into the edge band", () => {
  const shallow = autoscrollDelta(BOTTOM - AUTOSCROLL_EDGE + 1, TOP, BOTTOM);
  const deep = autoscrollDelta(BOTTOM - 1, TOP, BOTTOM);
  assert.ok(deep > shallow);
  assert.ok(shallow > 0);
});

test("the step is capped at the maximum even far past the edge", () => {
  // Dragging well outside the pane must not produce a runaway jump.
  assert.equal(autoscrollDelta(BOTTOM + 5000, TOP, BOTTOM), AUTOSCROLL_MAX_STEP);
  assert.equal(autoscrollDelta(TOP - 5000, TOP, BOTTOM), -AUTOSCROLL_MAX_STEP);
});

test("a pane shorter than two edge bands still picks a single direction", () => {
  // Bands would overlap at this height; the pointer's own half must win rather
  // than both branches firing.
  const shortTop = 0;
  const shortBottom = 40;
  const up = autoscrollDelta(2, shortTop, shortBottom);
  const down = autoscrollDelta(38, shortTop, shortBottom);
  assert.ok(up < 0);
  assert.ok(down > 0);
  assert.equal(autoscrollDelta(20, shortTop, shortBottom), 0);
});

test("a zero-height pane never scrolls", () => {
  assert.equal(autoscrollDelta(50, 50, 50), 0);
});
