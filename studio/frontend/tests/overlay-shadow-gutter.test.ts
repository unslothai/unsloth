// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The rail clips at its padding box, so with no padding under the bottom card
// that card lost its shadow, all of which falls below it: the llama.cpp toast
// was reported as "the bottom is cut off".
//
// Pinned here: the gutter is reserved under the cards and added on top of the
// band they may fill, so they stay put and a cap a few px short no longer
// slices one.

import assert from "node:assert/strict";
import test from "node:test";

import {
  STACK_SHADOW_GUTTER_BOTTOM,
  STACK_SHADOW_GUTTER_TOP,
  railBottomOffset,
  railCardsHeight,
  railMaxHeight,
  stackGeometry,
} from "../src/features/settings/stores/monitor-frame-store.ts";

const GUTTER = STACK_SHADOW_GUTTER_BOTTOM + STACK_SHADOW_GUTTER_TOP;

const INSET = 16;

test("the gutters clear the shadows the rail carries", () => {
  // Off the rendered blur: light is one level of white 8px below and gone 6px
  // above; dark is one level of #181818 16px below and 8px above.
  assert.ok(STACK_SHADOW_GUTTER_BOTTOM >= 16, "the shadow below is clipped");
  assert.ok(STACK_SHADOW_GUTTER_TOP >= 8, "the shadow above is clipped");
});

test("the rail's edge drops by the gutter, so the cards keep their inset", () => {
  // Offset plus padding is where the cards land.
  assert.equal(
    railBottomOffset(INSET) + STACK_SHADOW_GUTTER_BOTTOM,
    INSET,
    "the bottom card moved",
  );
  // A lifted placement keeps it too.
  assert.equal(railBottomOffset(148) + STACK_SHADOW_GUTTER_BOTTOM, 148);
});

test("the cap grows by both gutters, so the cards' band is unchanged", () => {
  for (const room of [0, 56, 137, 468]) {
    assert.equal(
      railMaxHeight(room) - GUTTER,
      room,
      "a gutter is being taken out of the cards",
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

test("the measured height discounts the gutters the rail carries", () => {
  const card = 137;
  assert.equal(railCardsHeight(card + GUTTER, GUTTER), card);
  // An empty rail is padding alone, and asks for nothing.
  assert.equal(railCardsHeight(GUTTER, GUTTER), 0);
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
  const measured = railCardsHeight(card + GUTTER, GUTTER);
  assert.deepEqual(
    stackGeometry([composer], W, H, measured, measured),
    stackGeometry([composer], W, H, card, card),
    "the gutter leaked into the placement",
  );
});
