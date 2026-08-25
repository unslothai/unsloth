// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  AUTOSCROLL_EDGE,
  AUTOSCROLL_MAX_STEP,
  autoscrollDelta,
  clipSpan,
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

test("an unclipped pane keeps its own span", () => {
  assert.deepEqual(clipSpan({ top: TOP, bottom: BOTTOM }, []), {
    top: TOP,
    bottom: BOTTOM,
  });
});

test("a pane clipped by an ancestor reports the visible span", () => {
  // The list pane runs past the bottom of the dialog body that scrolls it.
  const span = clipSpan({ top: 100, bottom: 900 }, [{ top: 0, bottom: 400 }]);
  assert.deepEqual(span, { top: 100, bottom: 400 });
});

test("clipping is what puts the edge back within the pointer's reach", () => {
  // Unclipped, the pane's bottom sits off-screen and a pointer at the bottom of
  // the window is nowhere near it, so a drag held there would never scroll.
  const pane = { top: 100, bottom: 2000 };
  const viewport = { top: 0, bottom: 800 };
  assert.equal(autoscrollDelta(790, pane.top, pane.bottom), 0);
  const span = clipSpan(pane, [viewport]);
  assert.ok(span);
  assert.ok(autoscrollDelta(790, span.top, span.bottom) > 0);
});

test("a pane scrolled entirely out of view has no span", () => {
  assert.equal(clipSpan({ top: 500, bottom: 900 }, [{ top: 0, bottom: 400 }]), null);
  // Touching edges are not visible either.
  assert.equal(clipSpan({ top: 400, bottom: 900 }, [{ top: 0, bottom: 400 }]), null);
});
