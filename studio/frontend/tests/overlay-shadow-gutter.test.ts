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
import { readFileSync } from "node:fs";
import test from "node:test";

const PROVIDER = readFileSync(
  new URL("../src/app/provider.tsx", import.meta.url),
  "utf8",
);

/** A `const NAME = <number>;` declaration in the provider. */
function constant(name: string): number {
  const found = PROVIDER.match(new RegExp(`const ${name} = (\\d+);`));
  assert.ok(found, `${name} is gone from the provider`);
  return Number(found[1]);
}

const GUTTER_BOTTOM = constant("STACK_SHADOW_GUTTER_BOTTOM");
const GUTTER_TOP = constant("STACK_SHADOW_GUTTER_TOP");

/** Where the cards sit, and the band they may fill, before the gutter. */
const CARDS_INSET = 16;
const CARDS_BAND_TRIM = 32;

test("the gutters clear the shadows the rail carries", () => {
  // Off the rendered blur: light is one level of white 8px below and gone 6px
  // above; dark is one level of #181818 16px below and 8px above.
  assert.ok(GUTTER_BOTTOM >= 16, "the shadow below is clipped");
  assert.ok(GUTTER_TOP >= 8, "the shadow above is clipped");
});

test("the rail's edge drops by the gutter, so the cards keep their inset", () => {
  // The rail is on the floor, so its bottom padding alone is the cards' inset.
  const rails = PROVIDER.match(/pointer-events-none fixed bottom-(\d+) right-4/g);
  assert.equal(rails?.length, 2, "a rail left its bottom-right corner");
  for (const rail of rails ?? []) {
    const edge = Number(rail.match(/bottom-(\d+)/)?.[1]);
    assert.equal(
      edge + GUTTER_BOTTOM,
      CARDS_INSET,
      "the bottom card moved off its inset",
    );
  }
});

test("the cap grows by both gutters, so the cards' band is unchanged", () => {
  // calc(100dvh - Npx), where N is the band's own trim less the gutter added
  // back. Anything larger spends the cards' room on the padding.
  const caps = PROVIDER.match(/max-h-\[calc\(100dvh_-_(\d+)px\)\]/g);
  assert.equal(caps?.length, 2, "a rail lost its cap");
  for (const cap of caps ?? []) {
    const trim = Number(cap.match(/_(\d+)px/)?.[1]);
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
});
