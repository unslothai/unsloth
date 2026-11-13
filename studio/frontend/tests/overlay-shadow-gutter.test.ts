// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The rail clips at its padding box, so it reserves a shadow gutter that its bottom padding
// and cap pay for, rather than shrinking the cards.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PROVIDER = readSrc("app/provider.tsx");

function constant(name: string): number {
  const found = PROVIDER.match(new RegExp(`const ${name} = (\\d+);`));
  assert.ok(found, `${name} is gone from the provider`);
  return Number(found[1]);
}

const GUTTER_BOTTOM = constant("STACK_SHADOW_GUTTER_BOTTOM");
const GUTTER_TOP = constant("STACK_SHADOW_GUTTER_TOP");
const GUTTER_LEFT = constant("STACK_SHADOW_GUTTER_LEFT");
const INSET_RIGHT = constant("STACK_CARD_INSET_RIGHT");

const CARDS_INSET = 16;
const CARDS_BAND_TRIM = 32;

test("the gutters clear the shadows the rail carries", () => {
  // Dark-mode shadow 0 8px 28px -6px reaches 22px sideways and 14px above a card.
  assert.ok(GUTTER_BOTTOM >= 16, "the shadow below is clipped");
  assert.ok(GUTTER_TOP >= 14, "the shadow above is clipped");
  assert.ok(GUTTER_LEFT >= 22, "the shadow to the left is clipped");
});

test("the rail's edge drops by the gutter, so the cards keep their inset", () => {
  // Bottom/right padding are the cards' insets: no gutter there, the screen edge clips.
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
  // Cap is 100dvh minus trim less gutters (and window chrome); past 100dvh the top goes off screen.
  const caps = PROVIDER.match(
    /max-h-\[(?:calc\()?100dvh(?:_-_(\d+)px\)|-var\(--studio-window-chrome-top,0px\)\))?\]/g,
  );
  assert.equal(caps?.length, 2, "a rail lost its cap");
  for (const cap of caps ?? []) {
    const trim = Number(cap.match(/_-_(\d+)px/)?.[1] ?? 0);
    assert.equal(
      trim,
      CARDS_BAND_TRIM - GUTTER_BOTTOM - GUTTER_TOP,
      "a gutter is being taken out of the cards' band",
    );
  }
});

test("the gutter is applied in px, not a rem utility", () => {
  // pb-4/pt-2 are rem-based, so at a non-16px root they would drift from the px insets.
  assert.match(PROVIDER, /paddingTop: STACK_SHADOW_GUTTER_TOP/);
  assert.match(PROVIDER, /paddingBottom: STACK_SHADOW_GUTTER_BOTTOM/);
  assert.match(PROVIDER, /paddingLeft: STACK_SHADOW_GUTTER_LEFT/);
  assert.match(PROVIDER, /paddingRight: STACK_CARD_INSET_RIGHT/);
});
