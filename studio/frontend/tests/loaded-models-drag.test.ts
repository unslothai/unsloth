// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clampToViewport,
  passedDragThreshold,
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
