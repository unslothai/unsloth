// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// UI font size has to move what separates the text too: a 20px label in a row
// padded for 12px reads as cramped. --ui-space-scale is the one multiplier
// every such length goes through.

const CSS = readSrc("index.css");
const SIDEBAR = readSrc("components/app-sidebar.tsx");
const PROVIDER = readSrc("app/provider.tsx");

const SCALED = /calc\([\d.]+(?:px|rem)\s*\*\s*var\(--ui-space-scale,\s*1\)\)/;

test("the spacing scale is the font-size preference, normalised at the default", () => {
  // --ui-font-scale is against a 16px base and the default is 15px, so
  // dividing by the default leaves the shipped spacing untouched.
  assert.match(
    CSS,
    /--ui-space-scale:\s*calc\(var\(--ui-font-scale, 1\) \/ 0\.9375\);/,
  );
  assert.match(CSS, /--ui-font-scale:\s*0\.9375;/);
});

test("every tailwind spacing utility goes through it", () => {
  // p-*, m-*, gap-* and size-* are all calc(var(--spacing) * n), so the step
  // scales the whole layout at once. The @theme passthrough is commented out,
  // so it is not a declaration.
  const declarations = (CSS.match(/--spacing:[^;]+;/g) ?? []).filter(
    (declaration) => !declaration.includes("var(--spacing)"),
  );
  assert.equal(declarations.length, 2, "light and dark each declare --spacing");
  for (const declaration of declarations) {
    assert.match(
      declaration,
      /--spacing:\s*calc\(0\.25rem \* var\(--ui-space-scale, 1\)\);/,
    );
  }
});

test("chat chrome that holds text scales, window chrome does not", () => {
  // The header band and the controls in it grow with their labels.
  for (const name of [
    "--studio-chat-header-height",
    "--studio-chat-control-height",
    "--studio-chat-header-padding-top",
  ]) {
    for (const source of [CSS, PROVIDER]) {
      const declaration = new RegExp(
        `${name}"?:\\s*"?(calc\\([^;\\n"]*\\)|[^;,\\n]+)`,
      ).exec(source);
      assert.ok(declaration, `${name} is gone`);
      assert.match(declaration[1], SCALED, `${name} must follow the UI scale`);
    }
  }

  // The title bar and the traffic lights belong to the OS window; a font size
  // preference must not move the UI off them.
  for (const name of [
    "--studio-desktop-titlebar-height",
    "--studio-content-top-inset",
  ]) {
    const declaration = new RegExp(`${name}":\\s*"([^"]+)"`).exec(PROVIDER);
    if (declaration) {
      assert.doesNotMatch(declaration[1], /--ui-space-scale/, name);
    }
  }
});

test("the sidebar's hand-set spacing follows the scale", () => {
  // The rows use one-off px values (gap-[8.5px], pl-[39px]), which is exactly
  // the spacing that used to stay put while the labels grew.
  const bare = [
    ...SIDEBAR.matchAll(
      /(?<![\w-])-?(?:p|px|py|pt|pb|pl|pr|m|mx|my|mt|mb|ml|mr|gap|gap-x|gap-y)-\[(\d*\.?\d+)px\]/g,
    ),
  ].filter((match) => Number(match[1]) > 1);
  assert.deepEqual(
    bare.map((match) => match[0]),
    [],
    "these sidebar paddings ignore the UI font size",
  );
});

test("fixed slots that hold scaled content scale with it", () => {
  // These heights live in JS, so the CSS variable cannot reach them. A slot
  // left at its 15px value clips or overlaps the row it holds.
  const HOOK = "useUiSpaceScale";
  const rows = readSrc("features/hub/catalog/models-catalog-rows.tsx");
  assert.ok(rows.includes(HOOK), "the virtualizer ignores the UI font size");
  assert.match(rows, /estimateSize: \(\) => slotHeight/);
  assert.match(rows, /height: `\$\{slotHeight\}px`/);
  // estimateSize is cached, so a new scale has to invalidate the sizes.
  assert.match(rows, /virtualizer\.measure\(\)/);

  const carousel = readSrc("features/hub/catalog/hub-section-row.tsx");
  assert.ok(carousel.includes(HOOK), "the carousel slot ignores it");
  assert.match(carousel, /itemHeight=\{cardHeight\}/);
  // The stride and the arrow's centre line are measured in JS against gap-4
  // and pt-2, which scale, so an unscaled stride stops short of the next card.
  const strip = readSrc("features/hub/catalog/card-carousel.tsx");
  assert.ok(strip.includes(HOOK), "the carousel stride ignores it");
  assert.match(strip, /const gapPx = CARD_GAP_PX \* scale;/);
  assert.match(strip, /const topPaddingPx = CAROUSEL_TOP_PADDING_PX \* scale;/);

  const lists = readSrc("features/hub/catalog/models-catalog-lists.tsx");
  assert.ok(lists.includes(HOOK), "the pinned grid ignores it");
});

test("the titlebar reserves room for its scaled controls", () => {
  const titlebar = readSrc("components/tauri/window-titlebar.tsx");
  // The drag region starts where this slot ends, so a fixed slot puts it on
  // top of the last scaled button.
  assert.match(titlebar, /calc\(7rem \* var\(--ui-space-scale, 1\)\)/);
  assert.match(PROVIDER, /calc\(112px \* var\(--ui-space-scale, 1\)\)/);
});
