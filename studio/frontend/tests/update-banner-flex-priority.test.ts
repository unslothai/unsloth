// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The overlay rail is height-capped by stackGeometry, and on the chat routes
// the composer publishes a box that makes the cap small. Nothing in the rail
// was shrink-0, so the cap came out of the update card and its release notes
// were painted over its own row of buttons.
//
// The rule pinned here: the notes are the only part of a card allowed to give
// up height, and they clip while doing it; the card floors at its buttons; the
// rail scrolls if that still does not fit.
//
// Read from the source: the node suite has no DOM to compute styles in.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function read(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

const TAURI = read("components/tauri/update-banner.tsx");
const WEB = read("components/web/update-banner.tsx");
const LLAMA = read("components/llama-update-banner.tsx");
const NOTES = read("components/update/release-notes-panel.tsx");
const PROVIDER = read("app/provider.tsx");
const STORE = read("features/settings/stores/monitor-frame-store.ts");

/** The class string opened by `anchor`, up to its closing quote. */
function classes(source: string, anchor: string): string {
  const at = source.indexOf(anchor);
  if (at === -1) {
    throw new Error(`${anchor} not found`);
  }
  return source.slice(at, source.indexOf('"', at + anchor.length));
}

const CARDS: ReadonlyArray<readonly [string, string]> = [
  ["tauri", TAURI],
  ["web", WEB],
];

for (const [name, source] of CARDS) {
  test(`the ${name} card's header cannot be compressed`, () => {
    assert.match(
      classes(source, "flex min-w-0 "),
      /\bshrink-0\b/,
      "the header shrinks, so the version line collides with the notes",
    );
  });

  test(`the ${name} card's action row cannot be compressed`, () => {
    const footer = classes(source, "mt-4 flex ");
    assert.match(
      footer,
      /\bshrink-0\b/,
      "the buttons shrink, so the notes are painted over them",
    );
    // Wrapping is how a narrow card copes; it is not compression.
    assert.match(footer, /\bflex-wrap\b/, "the buttons must still wrap");
  });

  test(`the ${name} card stops shrinking at its buttons`, () => {
    const stacked = classes(source, "pointer-events-auto flex ");
    // The floor is the header and the action row. It has to follow
    // --ui-font-scale, not be measured once at the default type size:
    // Settings > Appearance goes to 20px, where the action row wraps at every
    // card width and a 128px floor cuts the buttons in half. A fixed part plus
    // a scaled one, since only some of the card moves with the setting, and
    // scaling the whole box asked 256px where 209 was needed.
    assert.ok(
      !/\bmin-h-0\b/.test(stacked),
      "min-h-0 lets the rail squeeze the card to nothing",
    );
    assert.match(
      source,
      /min-h-\[calc\(\d+px\+\d+px\*var\(--ui-font-scale,1\)\)\]/,
      "the floor does not track the type size in the shape index.css uses",
    );
    assert.ok(
      !/12rem\*var\(--ui-font-scale/.test(source),
      "the whole box is being scaled again, which over-reserves the floor",
    );
    assert.ok(
      !/min-h-48|min-h-32/.test(source),
      "a leftover fixed floor still binds and still clips",
    );
  });
}

test("the desktop failure card does not shrink at all", () => {
  // It has no notes panel, so there is nothing in it to give up: shrinking it
  // only clips the diagnostics and the retry button.
  assert.match(
    TAURI,
    /showFailure\s*\n?\s*\? "shrink-0"/,
    "the failure card shares the notes-bearing card's floor, which is too low",
  );
});

test("the desktop failure card can scroll to its own diagnostics", () => {
  // The card is capped at the viewport and clips, and the rail cannot scroll
  // to what that cap hides, so the clipboard fallback needs its own scroller
  // or the report the reader is told to copy is the part that disappears.
  const region = classes(TAURI, "hover-scrollbar min-h-0 flex-1 ");
  assert.match(region, /\boverflow-y-auto\b/, "the report cannot be scrolled");
  assert.match(region, /\boverscroll-contain\b/);
  assert.ok(
    !/shrink-0[^"]*resize-none/.test(TAURI),
    "a shrink-0 textarea pushes itself past the card's cap again",
  );
});

test("the two update cards do not drift apart", () => {
  // One is the desktop card and one the browser card, but they are the same
  // card, so a fix applied to one and not the other is the bug coming back.
  // The headers are the same string; the roots share everything except the
  // floor, which the desktop card varies for its failure state.
  assert.equal(
    classes(TAURI, "flex min-w-0 "),
    classes(WEB, "flex min-w-0 "),
    "the headers differ between the desktop and browser cards",
  );
  // Both floors are dropped before comparing, the plain one and the narrow
  // variant: the two cards measure differently and are meant to differ here.
  const root = (source: string) =>
    classes(source, "pointer-events-auto flex ")
      .split(" ")
      .filter((rule) => !/^(max-\[\d+px\]:)?min-h-/.test(rule))
      .join(" ");
  assert.equal(
    root(TAURI),
    root(WEB),
    "the rail-facing roots differ between the desktop and browser cards",
  );
  // The action rows justify differently, so compare only what this fix pins.
  for (const rule of ["shrink-0", "flex-wrap"]) {
    assert.ok(
      classes(TAURI, "mt-4 flex ").includes(rule) &&
        classes(WEB, "mt-4 flex ").includes(rule),
      `one action row is missing ${rule}`,
    );
  }
});


test("the llama.cpp card keeps its full height in the rail", () => {
  // Nothing inside it can give up height, so squeezing it only mangles it.
  assert.match(
    classes(LLAMA, "pointer-events-auto w-[calc(100vw-2rem)]"),
    /\bshrink-0\b/,
  );
});

test("the notes panel clips whatever height it gives up", () => {
  assert.match(
    classes(NOTES, "mt-3 flex min-h-0 flex-col"),
    /\boverflow-hidden\b/,
    "the panel shrinks but its content still paints past the panel",
  );
  // The panel root is the clipper. The surface inside it keeps its intrinsic
  // height: a scroll container there collapses the expanded notes to nothing,
  // because their scroller is a flex-basis-0 child of it.
  assert.ok(
    !/overflow-hidden[^"]*rounded-\[14px\]/.test(NOTES),
    "the inner surface clips, which empties the expanded notes",
  );
});

test("the collapsed notes summary scrolls, like the expanded notes", () => {
  // The expanded notes already scrolled; the collapsed bullet list did not,
  // and the collapsed list is what the reported screenshot was showing.
  const summary = classes(NOTES, "hover-scrollbar min-h-0 flex-1 space-y-1");
  assert.match(summary, /\boverflow-y-auto\b/);
  assert.match(summary, /\boverscroll-contain\b/);
  const expanded = classes(NOTES, "hover-scrollbar max-h-64");
  assert.match(expanded, /\boverflow-y-auto\b/);
  assert.match(expanded, /\boverscroll-contain\b/);
});

/** The two rails' class strings, anchored on the corner they are pinned to. */
const RAIL_ANCHOR = '"pointer-events-none fixed bottom-0 right-4 ';

function rails(): string[] {
  const parts = PROVIDER.split(RAIL_ANCHOR);
  assert.equal(parts.length - 1, 2, "a rail left its bottom-right corner");
  return parts.slice(1);
}

test("the rail scrolls rather than spilling its cards", () => {
  for (const rail of rails()) {
    const rules = rail.slice(0, rail.indexOf('"'));
    // A cap without a scroller drops the overflow below the bottom of the
    // screen: at a large type size the two banner floors exceed the cap on
    // their own, and the cards under it cannot be reached.
    assert.match(rules, /\boverflow-y-auto\b/, "a capped rail spills its cards");
    // The scroller clips at the padding box, so without room reserved there the
    // cards lose their shadows; the negative margin puts the rail back where it
    // was.
    assert.match(rules, /\bpx-3\b/);
    assert.match(rules, /-mx-3/);
  }
});

// Reserved, not taken: the cards keep their band and the padding hangs below
// it. The rail sits on the floor and the bottom gutter carries the cards back
// up to 16px, so the cap grows by both gutters to pay for them.
test("the rail's block gutter costs the cards no room", () => {
  for (const rail of rails()) {
    const rules = rail.slice(0, rail.indexOf('"'));
    const style = rail.slice(rail.indexOf("style={{"), rail.indexOf("}}"));
    // 2rem for the cards' own band, less the 24px of gutter the rail adds
    // around them, so the cards keep exactly the band they had.
    assert.match(
      rules,
      /max-h-\[calc\(100dvh_-_8px\)\]/,
      "the gutter is being taken out of the cards' band",
    );
    // From the constants, not pb-4/pt-2: those are rem, so at any root size but
    // 16px the cards would drift off the corner.
    assert.match(
      style,
      /paddingTop: STACK_SHADOW_GUTTER_TOP/,
      "the top gutter can drift from the cap that pays for it",
    );
    assert.match(
      style,
      /paddingBottom: STACK_SHADOW_GUTTER_BOTTOM/,
      "the bottom gutter can drift from the cap that pays for it",
    );
    // Every surface offsets its shadow downwards, so flush against the clip
    // edge the bottom card loses all of it. A zero gutter is that bug again.
    assert.doesNotMatch(rules, /\bp[byt]-/, "a rem gutter is back on the rail");
  }
  // The gutter drops the rail's box to the floor and `-mx-3` put it 4px from the right
  // edge, so it spans the window's resize grips, which are under it on Tailwind's scale.
  // All eight: a narrow window spans the rail across the north and west targets too.
  const TITLEBAR = read("components/tauri/window-titlebar.tsx");
  // A z-index on the toolbar would read as protection and give none: it sits inside a
  // positioned, numbered header, which is a stacking context.
  const toolbar = TITLEBAR.slice(
    TITLEBAR.lastIndexOf("<div", TITLEBAR.indexOf('aria-label="Window controls"')),
    TITLEBAR.indexOf('aria-label="Window controls"'),
  );
  assert.doesNotMatch(
    toolbar,
    /zIndex:/,
    "the window-controls toolbar carries a z-index, which its header traps",
  );
  for (const grip of [
    "cursor-n-resize",
    "cursor-s-resize",
    "cursor-w-resize",
    "cursor-e-resize",
    "cursor-nw-resize",
    "cursor-ne-resize",
    "cursor-sw-resize",
    "cursor-se-resize",
  ]) {
    const target = TITLEBAR.slice(
      TITLEBAR.lastIndexOf("<div", TITLEBAR.indexOf(grip)),
      TITLEBAR.indexOf("/>", TITLEBAR.indexOf(grip)),
    );
    assert.doesNotMatch(
      target,
      /z-\[70\]/,
      `the ${grip} target is back under the overlay stack`,
    );
    assert.match(
      target,
      /zIndex: Z_LAYER\.WINDOW_RESIZE_EDGE/,
      `the ${grip} target does not take the named layer, so the rail covers it`,
    );
  }
});

// The rail was placed from JS for a while, lifting clear of the boxes the
// composer and the floating panels publish. Every input to that placement
// changes on its own, so the rail drifted to the middle and the top of the
// window. Anchored in CSS again; the floors above absorb a short window.
test("the rail is anchored to its corner, not placed from JS", () => {
  for (const rail of rails()) {
    const branch = rail.slice(0, rail.indexOf("style={{"));
    // Click-through in every state. It used to take pointer input while it
    // scrolled, which needed the JS that also placed it. The fold is reached by
    // wheeling over a card, whose nearest scrollable ancestor is the rail, or
    // by focus, which scrolls it into view.
    assert.doesNotMatch(
      branch,
      /pointer-events-auto/,
      "the rail takes pointer input again, which the placement paid for",
    );
  }
  // The offset and the cap are the two things the placement used to own, so
  // they are the two that must stay out of the render.
  for (const banned of [
    "useStackGeometry",
    "stackGeometry",
    "stack.bottom",
    "stack.maxHeight",
    "railBottomOffset",
    "railMaxHeight",
  ]) {
    assert.ok(
      !PROVIDER.includes(banned),
      `the rail is placed from JS again (${banned})`,
    );
  }
  // And the arithmetic it was placed by does not come back to the store, which
  // is now only a register of where the draggable panels are.
  for (const banned of [
    "stackBottomInset",
    "stackMaxHeight",
    "dodgeInset",
    "railCardsHeight",
  ]) {
    assert.ok(
      !STORE.includes(banned),
      `the dodge arithmetic is back in the frame store (${banned})`,
    );
  }
});
