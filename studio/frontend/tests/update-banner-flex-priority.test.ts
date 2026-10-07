// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only the notes may give up height (clipping); the card floors at its buttons; the rail scrolls.
// Read from source: the node suite has no DOM.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const TAURI = readSrc("components/tauri/update-banner.tsx");
const WEB = readSrc("components/web/update-banner.tsx");
const LLAMA = readSrc("components/llama-update-banner.tsx");
const LLAMA_CHANGELOG = readSrc(
  "components/update/llama-update-changelog-panel.tsx",
);
const NOTES_LAYOUT = readSrc("components/update/update-notes-layout.ts");
const NOTES = readSrc("components/update/release-notes-panel.tsx");
const PROVIDER = readSrc("app/provider.tsx");
const STORE = readSrc("features/settings/stores/monitor-frame-store.ts");

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
    assert.match(footer, /\bflex-wrap\b/, "the buttons must still wrap");
  });

  test(`the ${name} card stops shrinking at its buttons`, () => {
    const stacked = classes(source, "pointer-events-auto flex ");
    // The browser card names no height floor: a constant goes stale as spacing scales with font size,
    // so its automatic minimum (no min-h-0, no overflow-hidden) is used instead.
    assert.ok(
      !/\bmin-h-0\b/.test(stacked),
      "min-h-0 lets the rail squeeze the card to nothing",
    );
    if (name === "web") {
      const surface = classes(source, "relative flex max-h-[");
      for (const zeroesTheFloor of ["min-h-0", "overflow-hidden"]) {
        assert.ok(
          !surface.split(/\s+/).includes(zeroesTheFloor),
          `${zeroesTheFloor} puts the card's floor back to nothing`,
        );
      }
      assert.doesNotMatch(
        source,
        /min-h-\[/,
        "the card names a height again, which goes stale at the next type size",
      );
    } else {
      assert.match(
        source,
        /min-h-\[calc\(\d+px\+\d+px\*var\(--ui-font-scale,1\)\)\]/,
        "the floor does not track the type size in the shape index.css uses",
      );
    }
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

test("a card with no notes panel does not shrink at all", () => {
  for (const [name, source] of CARDS) {
    const stacked = classes(source, "pointer-events-auto flex ");
    assert.match(stacked, /\bshrink-0\b/, `the ${name} card can be squeezed`);
    assert.doesNotMatch(
      source,
      /["\s]min-h-\[calc\(/,
      `the ${name} card floors unconditionally again, around a card that may paint none of it`,
    );
    if (name !== "web") {
      assert.match(
        source,
        /has-\[\[data-slot=update-release-notes\]\]:min-h-\[calc\(/,
        `the ${name} card's floor is not gated on its notes panel`,
      );
    }
    assert.match(
      source,
      /has-\[\[data-slot=update-release-notes\]\]:shrink\b/,
      `the ${name} card cannot give up its notes' height once it has them`,
    );
  }
  assert.match(NOTES, /data-slot="update-release-notes"/);
});

test("a floored card paints all the height its slot reserves", () => {
  for (const [name, source] of [...CARDS, ["llama.cpp", LLAMA] as const]) {
    assert.match(
      classes(source, "relative flex max-h-[calc(100dvh_-_2rem"),
      /\bgrow\b/,
      `the ${name} card can be shorter than the slot it sits in`,
    );
  }
});

test("the desktop failure card can scroll to its own diagnostics", () => {
  // The rail cannot scroll to what the card's viewport cap hides, so the clipboard fallback needs
  // its own scroller.
  const region = classes(TAURI, "hover-scrollbar min-h-0 flex-1 ");
  assert.match(region, /\boverflow-y-auto\b/, "the report cannot be scrolled");
  assert.match(region, /\boverscroll-contain\b/);
  assert.ok(
    !/shrink-0[^"]*resize-none/.test(TAURI),
    "a shrink-0 textarea pushes itself past the card's cap again",
  );
});

test("the two update cards do not drift apart", () => {
  // Desktop and browser cards are the same card; a fix applied to only one is the bug returning.
  assert.equal(
    classes(TAURI, "flex min-w-0 "),
    classes(WEB, "flex min-w-0 "),
    "the headers differ between the desktop and browser cards",
  );
  const root = (source: string) =>
    classes(source, "pointer-events-auto flex ")
      .split(" ")
      .filter((rule) => !/(^|:)min-h-/.test(rule))
      .join(" ");
  assert.equal(
    root(TAURI),
    root(WEB),
    "the rail-facing roots differ between the desktop and browser cards",
  );
  for (const rule of ["shrink-0", "flex-wrap"]) {
    assert.ok(
      classes(TAURI, "mt-4 flex ").includes(rule) &&
        classes(WEB, "mt-4 flex ").includes(rule),
      `one action row is missing ${rule}`,
    );
  }
});

test("the llama.cpp card takes the desktop updater's floor only with its changelog open", () => {
  const slot = LLAMA.slice(
    LLAMA.indexOf("pointer-events-auto flex "),
    LLAMA.indexOf('data-testid="llama-update-banner"'),
  );
  for (const floor of [
    "min-h-[calc(117px+93px*var(--ui-font-scale,1))]",
    "min-h-[calc(24px+224px*var(--ui-font-scale,1))]",
  ]) {
    assert.ok(slot.includes(floor) && TAURI.includes(floor));
  }
  assert.match(slot, /max-\[383px\]:min-h-\[calc\(24px/);
  assert.match(
    slot,
    /changelogPanelOpen\s*\n?\s*\?\s*"min-h-\[calc\(117px/,
    "the floor is back on every state of the card, including the collapsed one",
  );
  assert.match(LLAMA, /\{changelogPanelOpen &&/);
  assert.match(
    slot,
    /"shrink-0"/,
    "with no notes to give up, the card must hold its height and let the rail scroll",
  );
  assert.doesNotMatch(classes(LLAMA, "pointer-events-auto flex "), /min-h-/);
  assert.doesNotMatch(slot, /\bmin-h-0\b/);
  assert.ok(LLAMA.includes("max-w-[calc(448px*var(--ui-space-scale,1))]"));
});

test("the llama.cpp changelog uses the desktop update notes layout", () => {
  for (const sharedClass of [
    "UPDATE_NOTES_ROOT_CLASS",
    "UPDATE_NOTES_SURFACE_CLASS",
    "UPDATE_NOTES_EXPANDED_SCROLL_CLASS",
    "UPDATE_NOTES_ITEM_CLASS",
    "UPDATE_NOTES_BULLET_CLASS",
    "UPDATE_NOTES_LEAD_CLASS",
    "UPDATE_NOTES_FOOTER_CLASS",
    "UPDATE_NOTES_LINK_CLASS",
  ]) {
    assert.ok(
      NOTES.includes(sharedClass) && LLAMA_CHANGELOG.includes(sharedClass),
      `${sharedClass} is not shared by both update panels`,
    );
  }
  assert.match(NOTES_LAYOUT, /\bmax-h-64\b/);
  assert.match(NOTES_LAYOUT, /\boverflow-y-auto\b/);
  assert.match(NOTES_LAYOUT, /\boverscroll-contain\b/);
});

test("the llama.cpp header and actions do not compress around its changelog", () => {
  assert.match(classes(LLAMA, "flex min-w-0 "), /\bshrink-0\b/);
  const footer = classes(LLAMA, "mt-4 flex shrink-0");
  assert.match(footer, /\bflex-wrap\b/);
  assert.match(footer, /\bshrink-0\b/);
});

test("the llama.cpp progress indicator is not a dead keyboard stop", () => {
  const progress = LLAMA.indexOf('role="progressbar"');
  assert.notEqual(progress, -1, "the progress indicator is missing");
  const openingTag = LLAMA.slice(progress, LLAMA.indexOf(">", progress));
  assert.doesNotMatch(openingTag, /\btabIndex=/);
});

test("the notes panel clips whatever height it gives up", () => {
  assert.match(
    classes(NOTES_LAYOUT, "mt-3 flex min-h-0 flex-1 flex-col"),
    /\boverflow-hidden\b/,
    "the panel shrinks but its content still paints past the panel",
  );
  // A scroll container on the inner surface collapses the notes, whose scroller is flex-basis-0.
  assert.ok(
    !/overflow-hidden[^"]*rounded-\[14px\]/.test(NOTES_LAYOUT),
    "the inner surface clips, which empties the expanded notes",
  );
});

test("the collapsed notes summary scrolls, like the expanded notes", () => {
  const summary = classes(NOTES, "hover-scrollbar min-h-0 flex-1 space-y-1");
  assert.match(summary, /\boverflow-y-auto\b/);
  assert.match(summary, /\boverscroll-contain\b/);
  const expanded = classes(NOTES_LAYOUT, "hover-scrollbar max-h-64");
  assert.match(expanded, /\boverflow-y-auto\b/);
  assert.match(expanded, /\boverscroll-contain\b/);
});

const RAIL_ANCHOR = '"pointer-events-none fixed bottom-0 right-0 ';

function rails(): string[] {
  const parts = PROVIDER.split(RAIL_ANCHOR);
  assert.equal(parts.length - 1, 2, "a rail left its bottom-right corner");
  return parts.slice(1);
}

test("the rail scrolls rather than spilling its cards", () => {
  for (const rail of rails()) {
    const rules = rail.slice(0, rail.indexOf('"'));
    // A cap without a scroller drops overflow off-screen at large type sizes.
    assert.match(
      rules,
      /\boverflow-y-auto\b/,
      "a capped rail spills its cards",
    );
    // The scroller clips at its padding box, so shadow room is reserved in px, never rem.
    assert.doesNotMatch(
      rules,
      /(^|\s)-?[mp][xlr]-/,
      "a rem gutter is back on the rail's inline axis",
    );
  }
});

function darkShadow(source: string): {
  y: number;
  blur: number;
  spread: number;
} {
  const seen = source.match(
    /dark:shadow-\[0_(\d+)px_(\d+)px_(-?\d+)px_/,
  );
  assert.ok(seen, "the card has no dark-mode shadow to size the gutter from");
  return { y: Number(seen[1]), blur: Number(seen[2]), spread: Number(seen[3]) };
}

function gutter(name: string): number {
  const seen = PROVIDER.match(new RegExp(`const ${name} = (\\d+);`));
  assert.ok(seen, `${name} is gone from the provider`);
  return Number(seen[1]);
}

// The dark-mode shadow reaches 22px; a smaller reserved gutter clips the fade mid-gradient.
test("the rail reserves enough room for the darkest card shadow", () => {
  for (const [name, source] of [...CARDS, ["llama", LLAMA]] as const) {
    const { y, blur, spread } = darkShadow(source);
    // Chromium paints the blur about its radius past the spread rect; negative spread pulls it in.
    const reach = blur + spread;
    assert.ok(
      gutter("STACK_SHADOW_GUTTER_LEFT") >= reach,
      `the ${name} card's halo is cut off on the left`,
    );
    assert.ok(
      gutter("STACK_SHADOW_GUTTER_TOP") >= reach - y,
      `the ${name} card's halo is cut off above it`,
    );
  }
});

// The cap grows by both gutters so the reserved padding costs the cards no room.
test("the rail's block gutter costs the cards no room", () => {
  for (const rail of rails()) {
    const rules = rail.slice(0, rail.indexOf('"'));
    const style = rail.slice(rail.indexOf("style={{"), rail.indexOf("}}"));
    assert.match(
      rules,
      /max-h-\[(?:calc\()?100dvh(?:_-_\d+px\)|-var\(--studio-window-chrome-top,0px\)\))?\]/,
      "the rail lost the cap that pays for its gutters",
    );
    // From the constants, not rem utilities, or cards drift off the corner at other root sizes.
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
    // Asymmetric: a left gutter where a cut halo shows; the right edge is the screen.
    assert.match(
      style,
      /paddingLeft: STACK_SHADOW_GUTTER_LEFT/,
      "the left gutter is back on a rem utility, or gone",
    );
    assert.match(
      style,
      /paddingRight: STACK_CARD_INSET_RIGHT/,
      "the cards' right inset is back on a rem utility, or gone",
    );
    // Shadows offset downward, so a zero bottom gutter clips the bottom card's shadow.
    assert.doesNotMatch(rules, /\bp[byt]-/, "a rem gutter is back on the rail");
  }
  // The rail box spans the window resize grips, which sit under it on Tailwind's z-scale.
  const TITLEBAR = readSrc("components/tauri/window-titlebar.tsx");
  // A z-index on the toolbar gives no protection: its header is a stacking context.
  const toolbar = TITLEBAR.slice(
    TITLEBAR.lastIndexOf(
      "<div",
      TITLEBAR.indexOf('aria-label="Window controls"'),
    ),
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

// JS placement drifted because every input changes independently; keep it anchored in CSS.
test("the rail is anchored to its corner, not placed from JS", () => {
  for (const rail of rails()) {
    const branch = rail.slice(0, rail.indexOf("style={{"));
    // Click-through in every state; the fold is reached by wheel over a card or by focus.
    assert.doesNotMatch(
      branch,
      /pointer-events-auto/,
      "the rail takes pointer input again, which the placement paid for",
    );
  }
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
