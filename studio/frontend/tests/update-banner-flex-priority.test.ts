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
    // The floor is the card with its notes closed, measured in a browser: 8rem
    // for one row of actions, 12rem once the card is narrow enough to wrap them
    // onto two. min-h-0 was the old floor and it is no floor at all.
    assert.ok(
      !/\bmin-h-0\b/.test(stacked),
      "min-h-0 lets the rail squeeze the card to nothing",
    );
    assert.match(
      source,
      /min-h-48[^"]*min-\[480px\]:min-h-32/,
      "one floor for both widths clips the wrapped action row",
    );
  });
}

test("the desktop failure card does not shrink at all", () => {
  // It has no notes panel, so there is nothing in it to give up: shrinking it
  // only clips the diagnostics and the retry button.
  assert.match(
    TAURI,
    /showFailure \? "shrink-0" : "min-h-48 min-\[480px\]:min-h-32"/,
    "the failure card shares the notes-bearing card's floor, which is too low",
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
  const root = (source: string) =>
    classes(source, "pointer-events-auto flex ")
      .split(" ")
      .filter((rule) => !rule.includes("min-h-32") && rule !== "min-h-48")
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

test("the rail scrolls rather than spilling its cards", () => {
  const rails = PROVIDER.split("fixed right-4 z-[9998]");
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
    // Sideways only. This node is the one useStackGeometry measures, so
    // vertical padding is counted into scrollHeight and the stack asks for room
    // it does not occupy: a card that fits under the composer then measures as
    // one that does not, and the rail lifts over it. The cap has to stay on
    // this same node too, since measure() lifts it here to read the natural
    // height; on a parent, the read would be the placement's own output.
    assert.ok(
      !/\bp-3\b/.test(rules) && !/\bpy-3\b/.test(rules),
      "vertical padding on the measured node inflates the measured height",
    );
  }
});

// pointer-events-none takes the rail's scrollbar with it, and only the cards
// opt back in, so a scrolling rail nobody can drag hides the cards under the
// fold. It is click-through only while there is nothing to scroll to.
test("a scrolling rail takes pointer input, a fitting one stays click-through", () => {
  const rails = PROVIDER.split("fixed right-4 z-[9998]");
  assert.equal(rails.length - 1, 2, "the rail count changed");
  for (const rail of rails.slice(1)) {
    const branch = rail.slice(0, rail.indexOf("style={{"));
    assert.match(
      branch,
      /stack\.overflowing\s*\?\s*"pointer-events-auto"\s*:\s*"pointer-events-none"/,
      "the rail is click-through unconditionally",
    );
  }
  // Read from the capped box. Comparing the natural height against the cap
  // instead reports every stack whose cards are giving up height, which is
  // most of them, and hands the rail pointer input it does not need.
  assert.match(
    STORE,
    /const scrolls = node\.scrollHeight > node\.clientHeight;/,
    "the overflow flag is not measured on the capped box",
  );
});
