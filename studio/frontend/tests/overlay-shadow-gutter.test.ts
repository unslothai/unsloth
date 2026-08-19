// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The bottom-right overlay rail clips at its padding box. With no padding under
// the bottom card that card lost its shadow, every bit of which falls below it,
// and the llama.cpp update toast was reported as "the bottom is cut off".
//
// The rule pinned here: the gutter is reserved under the cards and added on top
// of the band they may fill, so they stay where they were and a cap a few px
// short of a card no longer slices one.

import assert from "node:assert/strict";
import test from "node:test";

import {
  STACK_SHADOW_GUTTER,
  railBottomOffset,
  railCardsHeight,
  railMaxHeight,
  stackGeometry,
} from "../src/features/settings/stores/monitor-frame-store.ts";

const INSET = 16;

test("the gutter clears the deepest shadow the rail carries", () => {
  // Dark theme, 0 8px 28px -6px, reaches 8 + 28/2 - 6 below the card. Light
  // reaches 4; neither reaches above the top edge.
  assert.ok(
    STACK_SHADOW_GUTTER >= 8 + 28 / 2 - 6,
    "the dark theme's shadow is still clipped",
  );
});

test("the rail's edge drops by the gutter, so the cards keep their inset", () => {
  // Offset plus padding is where the cards land.
  assert.equal(
    railBottomOffset(INSET) + STACK_SHADOW_GUTTER,
    INSET,
    "the bottom card moved",
  );
  // A lifted placement keeps it too.
  assert.equal(railBottomOffset(148) + STACK_SHADOW_GUTTER, 148);
});

test("the cap grows by the gutter, so the cards' band is unchanged", () => {
  for (const room of [0, 56, 137, 468]) {
    assert.equal(
      railMaxHeight(room) - STACK_SHADOW_GUTTER,
      room,
      "the gutter is being taken out of the cards",
    );
  }
});

test("the clip box holds a card the cap is a little short of", () => {
  const card = 137;
  // A cap a few px short of the card used to slice its bottom corners off.
  assert.ok(
    railMaxHeight(card - 10) >= card,
    "the rail still slices a card the cap is 10px short of",
  );
  // Not a licence to overflow by any amount: a cramped rail still scrolls.
  assert.ok(railMaxHeight(card - 40) < card);
});

test("the measured height discounts the gutter the rail carries", () => {
  const card = 137;
  assert.equal(
    railCardsHeight(card + STACK_SHADOW_GUTTER, STACK_SHADOW_GUTTER),
    card,
  );
  // An empty rail is padding alone, and asks for nothing.
  assert.equal(railCardsHeight(STACK_SHADOW_GUTTER, STACK_SHADOW_GUTTER), 0);
  // Never negative, whatever scrollHeight rounds to.
  assert.equal(railCardsHeight(15, 16), 0);
});

test("a discounted measurement places the rail as an unpadded one did", () => {
  const W = 1280;
  const H = 720;
  const card = 137;
  // The composer, docked under a thread with turns.
  const composer = {
    left: 220,
    top: H - 140,
    right: 1180,
    bottom: H,
    coverable: true,
  };
  const measured = railCardsHeight(
    card + STACK_SHADOW_GUTTER,
    STACK_SHADOW_GUTTER,
  );
  assert.deepEqual(
    stackGeometry([composer], W, H, measured, measured),
    stackGeometry([composer], W, H, card, card),
    "the gutter leaked into the placement",
  );
});
