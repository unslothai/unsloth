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

// Covering the composer is licensed per card, and the licence is carried on
// the card rather than inferred in the store, so a new overlay added to the
// rail is persistent until someone says otherwise. The loaded models indicator
// and the download panel deliberately do not carry it: they are the last
// children, so they are what would land on Send.
test("only the dismissible cards licence covering the composer", () => {
  for (const [name, source] of CARDS) {
    assert.match(
      source,
      /data-overlay-dismissible="true"/,
      `the ${name} card lost its dismissible marker, so the stack will never cover`,
    );
  }
  // The llama.cpp card carries the licence CONDITIONALLY: its dismiss button
  // goes away for the length of an update, and a card that cannot be got rid
  // of must not be one the stack parks on Send.
  assert.match(
    LLAMA,
    /data-overlay-dismissible=\{applying \? undefined : "true"\}/,
    "the llama card licences covering the composer while it is mid-update",
  );
  assert.match(
    LLAMA,
    /\{applying \? null : \(\s*\n\s*<button/,
    "the dismiss button is no longer the thing the licence is tied to",
  );
  const indicator = read("features/loaded-models/loaded-models-indicator.tsx");
  const downloads = read(
    "features/hub/download-manager/download-manager-panel.tsx",
  );
  for (const [name, source] of [
    ["loaded models indicator", indicator],
    ["download panel", downloads],
  ] as const) {
    assert.ok(
      !/data-overlay-dismissible/.test(source),
      `the ${name} is persistent and must not licence covering the composer`,
    );
  }
  assert.match(
    STORE,
    /child\.hasAttribute\("data-overlay-dismissible"\)/,
    "the store no longer asks the cards whether it may cover",
  );
  // Measured up from the corner, so the run that stops a cover is the one that
  // would land on the composer, not any persistent card anywhere in the stack.
  // Over `cards`, the rail's in-flow children: a dragged loaded models card is
  // `position: fixed` somewhere else and lands on nothing.
  assert.match(
    STORE,
    /for \(let i = cards\.length - 1; i >= 0; i -= 1\)/,
    "the persistent run is no longer counted from the bottom of the stack",
  );
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

test("the rail scrolls rather than spilling its cards", () => {
  const rails = PROVIDER.split('"fixed right-4 ');
  // The desktop rail and the browser rail.
  assert.equal(rails.length - 1, 2, "the rail count changed");
  for (const rail of rails.slice(1)) {
    const rules = rail.slice(0, rail.indexOf('"'));
    assert.match(rules, /\boverflow-y-auto\b/, "a capped rail clips its cards");
    // The scroller clips at the padding box, so without room reserved there the
    // cards lose their shadows; the negative margin puts the rail back where it
    // was.
    assert.match(rules, /\bpx-3\b/);
    assert.match(rules, /-mx-3/);
    // scrollHeight spans that padding, so discount it or the stack asks for room
    // it does not occupy and lifts over the composer for nothing.
    assert.match(
      STORE,
      /const natural = railCardsHeight\(node\.scrollHeight, gutter\)/,
      "the natural height counts the rail's own padding as cards",
    );
    // Both totals: the squeezed one is what `overflowing` compares against the
    // cards' cap, and the per-card readings are border boxes.
    assert.match(
      STORE,
      /const collapsed = railCardsHeight\(node\.scrollHeight, gutter\)/,
      "the squeezed height counts the rail's own padding as cards",
    );
  }
});

// Reserved, not taken: the cards keep their band and the padding hangs below
// it. A negative margin cannot do this -- `bottom` anchors the margin edge, so
// the padding would move the cards instead.
test("the rail's block gutter costs the cards no room", () => {
  const rails = PROVIDER.split('"fixed right-4 ');
  assert.equal(rails.length - 1, 2, "the rail count changed");
  for (const rail of rails.slice(1)) {
    const rules = rail.slice(0, rail.indexOf('"'));
    const style = rail.slice(rail.indexOf("style={{"), rail.indexOf("}}"));
    assert.match(
      style,
      /bottom: railBottomOffset\(stack\.bottom\)/,
      "the rail's own edge is anchored where its cards belong",
    );
    assert.match(
      style,
      /maxHeight: railMaxHeight\(stack\.maxHeight\)/,
      "the gutter is spent on the cards' cap",
    );
    // From the same constants, not pb-4/pt-2: those are rem, so at any root size
    // but 16px the padding and the compensation drift apart.
    assert.match(
      style,
      /paddingTop: STACK_SHADOW_GUTTER_TOP/,
      "the top gutter can drift from the cap that pays for it",
    );
    assert.match(
      style,
      /paddingBottom: STACK_SHADOW_GUTTER_BOTTOM/,
      "the bottom gutter can drift from the offset that pays for it",
    );
    // Every surface offsets its shadow downwards, so flush against the clip
    // edge the bottom card loses all of it. A zero gutter is that bug again.
    assert.doesNotMatch(rules, /\bp[byt]-/, "a rem gutter is back on the rail");
  // The gutter drops the rail's box to the floor and `-mx-3` put it 4px from the right
  // edge, so a scrolling rail lands on the window's resize grips, which are under it on
  // Tailwind's scale. All eight: a narrow window spans the rail across the north and west
  // targets too.
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
      `the ${grip} target does not take the named layer, so a scrolling rail covers it`,
    );
  }
  }
});

// pointer-events-none takes the rail's scrollbar with it, and only the cards
// opt back in, so a scrolling rail nobody can drag hides the cards under the
// fold. It is click-through only while there is nothing to scroll to.
test("a scrolling rail takes pointer input, a fitting one stays click-through", () => {
  const rails = PROVIDER.split('"fixed right-4 ');
  assert.equal(rails.length - 1, 2, "the rail count changed");
  for (const rail of rails.slice(1)) {
    const branch = rail.slice(0, rail.indexOf("style={{"));
    assert.match(
      branch,
      /stack\.overflowing\s*\?\s*"pointer-events-auto"\s*:\s*"pointer-events-none"/,
      "the rail is click-through unconditionally",
    );
  }
  // Derived from the placement, never read back off the node. A DOM reading
  // latches: the stack scrolls for a frame, the placement then changes to one
  // that fits, and nothing resizes afterwards to correct the flag, so a rail
  // with nothing to scroll to keeps the pointer input it took. Compared
  // against the height the cards collapse to, not their natural height nor the
  // floor the placement asks for: a card clipped below its floor is not a fold
  // to scroll to.
  assert.match(
    STORE,
    /overflowing: collapsedRoom > geometry\.maxHeight/,
    "the overflow flag is not derived from the placement",
  );
  assert.ok(!/setOverflowing/.test(STORE), "a latched DOM reading is back");
  // The one place the flag can still latch is the empty-stack return, which
  // publishes nothing and leaves. Before the rail carried a gutter that cost
  // nothing: an empty rail was a zero-height box, so a stale "it scrolls"
  // bought pointer input over no pixels. The gutter gives that box a height,
  // so the corner would answer clicks with nothing in it. Every reading the
  // branch leaves behind has to be cleared, the DOM one included.
  const emptied = STORE.slice(
    STORE.indexOf("node.childElementCount === 0"),
    STORE.indexOf('node.style.transition = "none"'),
  );
  assert.ok(emptied.length > 0, "the empty-stack branch moved");
  for (const cleared of [
    "setNeededRoom",
    "setFloorRoom",
    "setCollapsedRoom",
    "setPersistentTail",
    "setDomOverflowing",
  ]) {
    assert.match(
      emptied,
      new RegExp(`${cleared}\\(`),
      `an emptied rail keeps its ${cleared} reading, so it still holds the corner`,
    );
  }
  // The probe writes max-height twice and reads scrollHeight between the
  // writes. transition-property: all reaches the rail, so unsuppressed each
  // write starts a transition, and a transition computes its start value
  // until the timeline advances: the box computes 0px while its inline style
  // reads back as the cap, and three whole cards lay out below it.
  const suppressed = STORE.indexOf('node.style.transition = "none"');
  const probe = STORE.slice(
    suppressed,
    STORE.indexOf("setNeededRoom", suppressed),
  );
  assert.ok(
    probe.length > 0 && /node\.style\.maxHeight = "none"/.test(probe),
    "the height probe is not run with transitions suppressed",
  );
  assert.match(
    STORE,
    /void node\.scrollHeight;\s*\n\s*node\.style\.transition = eased;/,
    "the restored cap is handed back to a transition",
  );
});
