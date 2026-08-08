// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clampSize,
  clampToViewport,
  passedDragThreshold,
  resizeFromTopLeft,
} from "../src/features/loaded-models/use-drag-position.ts";

const VIEWPORT = { width: 1440, height: 900 };
const CARD = { width: 268, height: 160 };

// The collapsed pill is both the drag handle and the button that expands the
// card, so a press that barely moves has to stay a click.
test("a press that barely moves is a click, not a drag", () => {
  assert.equal(passedDragThreshold(0, 0), false);
  assert.equal(passedDragThreshold(2, 2), false);
});

test("a press that moves is a drag", () => {
  assert.equal(passedDragThreshold(10, 0), true);
  assert.equal(passedDragThreshold(0, -10), true);
  assert.equal(passedDragThreshold(3, 3), true);
});

test("a position inside the viewport is left alone", () => {
  const position = { left: 400, top: 300 };
  assert.deepEqual(
    clampToViewport(position, CARD.width, CARD.height, VIEWPORT),
    position,
  );
});

test("dragging past an edge keeps the whole card on screen", () => {
  const offRight = clampToViewport(
    { left: 5000, top: 5000 },
    CARD.width,
    CARD.height,
    VIEWPORT,
  );
  assert.equal(offRight.left, VIEWPORT.width - CARD.width - 8);
  assert.equal(offRight.top, VIEWPORT.height - CARD.height - 8);

  const offTopLeft = clampToViewport(
    { left: -500, top: -500 },
    CARD.width,
    CARD.height,
    VIEWPORT,
  );
  assert.deepEqual(offTopLeft, { left: 8, top: 8 });
});

// Expanding a pill that was dragged to the bottom edge grows the card
// downwards, which is what the re-clamp on resize has to catch.
test("a card taller than the space left is pulled back up", () => {
  const grown = clampToViewport(
    { left: 100, top: VIEWPORT.height - 60 },
    CARD.width,
    320,
    VIEWPORT,
  );
  assert.equal(grown.top, VIEWPORT.height - 320 - 8);
});

test("a window smaller than the card still leaves it reachable", () => {
  const tiny = { width: 200, height: 120 };
  const clamped = clampToViewport(
    { left: 400, top: 400 },
    CARD.width,
    CARD.height,
    tiny,
  );
  assert.deepEqual(clamped, { left: 8, top: 8 });
});

/** The card where it rests by default: bottom-right, inset 16. */
const RESTING = {
  left: VIEWPORT.width - 16 - CARD.width,
  top: VIEWPORT.height - 16 - CARD.height,
  width: CARD.width,
  height: CARD.height,
};

// The whole point of the top-left grip: the corner it is anchored to has no
// room, so growing has to happen on the other side.
test("resizing holds the bottom-right corner still", () => {
  const grown = resizeFromTopLeft(RESTING, -120, -80);
  assert.equal(grown.size.width, CARD.width + 120);
  assert.equal(grown.size.height, CARD.height + 80);
  assert.equal(
    grown.position.left + grown.size.width,
    RESTING.left + RESTING.width,
  );
  assert.equal(
    grown.position.top + grown.size.height,
    RESTING.top + RESTING.height,
  );
});

test("dragging the grip inwards shrinks the card", () => {
  const shrunk = resizeFromTopLeft(RESTING, 40, 20);
  assert.equal(shrunk.size.width, CARD.width - 40);
  assert.equal(shrunk.size.height, CARD.height - 20);
  assert.equal(shrunk.position.left, RESTING.left + 40);
});

test("a resize cannot push the card past the top-left edge", () => {
  const huge = resizeFromTopLeft(RESTING, -5000, -5000);
  assert.equal(huge.position.left, 8);
  assert.equal(huge.position.top, 8);
  // Still anchored, so the box is exactly the room that was available.
  assert.equal(huge.size.width, RESTING.left + RESTING.width - 8);
  assert.equal(huge.size.height, RESTING.top + RESTING.height - 8);
});

test("a resize cannot shrink the card below its floor", () => {
  const tiny = resizeFromTopLeft(RESTING, 5000, 5000);
  assert.equal(tiny.size.width, 216);
  assert.equal(tiny.size.height, 116);
  // The floor wins, and the held corner still does not move.
  assert.equal(
    tiny.position.left + tiny.size.width,
    RESTING.left + RESTING.width,
  );
});

test("a floor larger than the room left still returns the floor", () => {
  assert.deepEqual(clampSize({ width: 300, height: 300 }, 10, 10), {
    width: 216,
    height: 116,
  });
});
