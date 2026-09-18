// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The rail clips at its padding box, so it reserves a gutter around its cards
// or their shadows are cut off and a cap a few px short slices a card's corners
// (#9246). The gutter must not be taken out of the cards: the rail sits on the
// floor and its bottom padding carries them back up to their inset, and its cap
// grows by both gutters to pay for them.
//
// Arithmetic in CSS rather than in JS, since the rail is anchored and not
// placed, so this reads the source: the node suite has no DOM to compute in.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PROVIDER = readSrc("app/provider.tsx");

/** A `const NAME = <number>;` declaration in the provider. */
function constant(name: string): number {
  const found = PROVIDER.match(new RegExp(`const ${name} = (\\d+);`));
  assert.ok(found, `${name} is gone from the provider`);
  return Number(found[1]);
}

const GUTTER_BOTTOM = constant("STACK_SHADOW_GUTTER_BOTTOM");
const GUTTER_TOP = constant("STACK_SHADOW_GUTTER_TOP");
const GUTTER_LEFT = constant("STACK_SHADOW_GUTTER_LEFT");
const INSET_RIGHT = constant("STACK_CARD_INSET_RIGHT");

/** Where the cards sit, and the band they may fill, before the gutter. */
const CARDS_INSET = 16;
const CARDS_BAND_TRIM = 32;

test("the gutters clear the shadows the rail carries", () => {
  // The dark-mode shadow is the deepest: 0 8px 28px -6px reaches 22px to a
  // card's side and 14px above it. Both were short, so the halo ended flat.
  assert.ok(GUTTER_BOTTOM >= 16, "the shadow below is clipped");
  assert.ok(GUTTER_TOP >= 14, "the shadow above is clipped");
  assert.ok(GUTTER_LEFT >= 22, "the shadow to the left is clipped");
});

test("the rail's edge drops by the gutter, so the cards keep their inset", () => {
  // The rail is in the corner, so its bottom and right padding are the cards'
  // insets. No gutter there: the clip is the screen edge.
  const rails = PROVIDER.match(
    /pointer-events-none fixed bottom-(\d+) right-(\d+)/g,
  );
  assert.equal(rails?.length, 2, "a rail left its bottom-right corner");
  for (const rail of rails ?? []) {
    const floor = Number(rail.match(/bottom-(\d+)/)?.[1]);
    const edge = Number(rail.match(/right-(\d+)/)?.[1]);
    assert.equal(
      floor + GUTTER_BOTTOM,
      CARDS_INSET,
      "the bottom card moved off its inset",
    );
    assert.equal(
      edge + INSET_RIGHT,
      CARDS_INSET,
      "the cards moved off their right inset",
    );
  }
});

test("the cap grows by both gutters, so the cards' band is unchanged", () => {
  // calc(100dvh - Npx), N being the band's trim less the gutters added back, so
  // a plus once they outgrow it. Anything smaller spends the cards' own room.
  const caps = PROVIDER.match(/max-h-\[calc\(100dvh_([-+])_(\d+)px\)\]/g);
  assert.equal(caps?.length, 2, "a rail lost its cap");
  for (const cap of caps ?? []) {
    const [, sign, size] = cap.match(/100dvh_([-+])_(\d+)px/) ?? [];
    const trim = (sign === "-" ? 1 : -1) * Number(size);
    assert.equal(
      trim,
      CARDS_BAND_TRIM - GUTTER_BOTTOM - GUTTER_TOP,
      "a gutter is being taken out of the cards' band",
    );
  }
});

test("the gutter is applied in px, not a rem utility", () => {
  // pb-4/pt-2 resolve through --spacing in rem, so at any root but 16px the
  // padding and the inset above would disagree and the cards would drift.
  assert.match(PROVIDER, /paddingTop: STACK_SHADOW_GUTTER_TOP/);
  assert.match(PROVIDER, /paddingBottom: STACK_SHADOW_GUTTER_BOTTOM/);
  assert.match(PROVIDER, /paddingLeft: STACK_SHADOW_GUTTER_LEFT/);
  assert.match(PROVIDER, /paddingRight: STACK_CARD_INSET_RIGHT/);
});
