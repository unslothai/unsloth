// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The card wore menu-soft-surface's inset edge ring, which the update banners
// do not. Read from the source: the node suite has no DOM to compute styles in.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const CSS = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
const INDICATOR = readFileSync(
  new URL(
    "../src/features/loaded-models/loaded-models-indicator.tsx",
    import.meta.url,
  ),
  "utf8",
);
const BANNER = readFileSync(
  new URL("../src/components/llama-update-banner.tsx", import.meta.url),
  "utf8",
);

function surface(source: string, anchor: string): string {
  const at = source.indexOf(anchor);
  assert.notEqual(at, -1, `${anchor} not found`);
  return source.slice(at, source.indexOf('"', at));
}

/** The CSS rule opened by `selector`, up to its closing brace. */
function rule(selector: string): string {
  const at = CSS.indexOf(selector);
  assert.notEqual(at, -1, `${selector} not found`);
  return CSS.slice(at, CSS.indexOf("}", at));
}

test("both card states drop the edge ring", () => {
  const pill = surface(INDICATOR, "menu-soft-surface menu-soft-edgeless");
  assert.match(pill, /menu-soft-edgeless/);
  // Two: the collapsed pill and the expanded card.
  assert.equal(
    INDICATOR.split("menu-soft-edgeless").length - 1,
    2,
    "the pill and the card are one surface, so both or neither",
  );
  assert.doesNotMatch(
    INDICATOR,
    /"menu-soft-surface pointer-events-auto/,
    "a bare menu-soft-surface still carries the ring",
  );
});

// The modifier keeps the drop shadow. Dropping both would flatten the card
// into the page.
test("the modifier removes only the inset, not the shadow", () => {
  const modifier = rule(".menu-soft-surface.menu-soft-edgeless");
  assert.doesNotMatch(modifier, /inset/);
  assert.match(modifier, /var\(--menu-soft-offset-y\)/);
  assert.match(modifier, /var\(--menu-soft-blur\)/);
  assert.match(modifier, /var\(--menu-soft-spread\)/);
  assert.match(modifier, /var\(--menu-soft-shadow\)/);
});

// Written against the vars rather than literals, so the dark override at
// .dark .menu-soft-surface still reaches it.
test("the modifier hardcodes neither theme", () => {
  const modifier = rule(".menu-soft-surface.menu-soft-edgeless");
  assert.doesNotMatch(modifier, /rgba|#[0-9a-f]{3}/i);
});

// The point of the change: the card and the banners are the same surface.
// menu-soft-surface's vars already carry the banner's geometry, so once the
// inset is gone the two agree. If the banner is restyled, this fails.
test("the shared vars still match the update banner's shadow", () => {
  assert.match(BANNER, /shadow-\[0_2px_8px_-2px_rgba\(0,0,0,0\.16\)\]/);
  assert.match(BANNER, /dark:shadow-\[0_8px_28px_-6px_rgba\(0,0,0,0\.28\)\]/);
  const light = rule(".menu-soft-surface,");
  assert.match(light, /--menu-soft-shadow: rgba\(0, 0, 0, 0\.16\)/);
  assert.match(light, /--menu-soft-offset-y: 2px/);
  assert.match(light, /--menu-soft-blur: 8px/);
  assert.match(light, /--menu-soft-spread: -2px/);
  const dark = rule(".dark .menu-soft-surface,");
  assert.match(dark, /--menu-soft-shadow: rgba\(0, 0, 0, 0\.28\)/);
  assert.match(dark, /--menu-soft-offset-y: 8px/);
  assert.match(dark, /--menu-soft-blur: 28px/);
  assert.match(dark, /--menu-soft-spread: -6px/);
});

// The banner paints bg-white/dark:bg-card, the card bg-popover. Same colour in
// every theme, so the surfaces match; this pins that they stay equal.
test("popover and card resolve to the same colour in every theme", () => {
  const popover = [...CSS.matchAll(/^\t*--popover:\s*([^;]+);/gm)].map((m) =>
    m[1].trim(),
  );
  const card = [...CSS.matchAll(/^\t*--card:\s*([^;]+);/gm)].map((m) =>
    m[1].trim(),
  );
  assert.ok(popover.length > 0);
  assert.deepEqual(popover, card);
});
