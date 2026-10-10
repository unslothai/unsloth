// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read from source: the node suite has no DOM to compute styles.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const INDICATOR = readSrc("features/loaded-models/loaded-models-indicator.tsx");
const BANNER = readSrc("components/llama-update-banner.tsx");

function surface(source: string, anchor: string): string {
  const at = source.indexOf(anchor);
  assert.notEqual(at, -1, `${anchor} not found`);
  return source.slice(at, source.indexOf('"', at));
}

function rule(selector: string): string {
  const at = CSS.indexOf(selector);
  assert.notEqual(at, -1, `${selector} not found`);
  return CSS.slice(at, CSS.indexOf("}", at));
}

test("both card states drop the edge ring", () => {
  const pill = surface(INDICATOR, "menu-soft-surface menu-soft-edgeless");
  assert.match(pill, /menu-soft-edgeless/);
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

test("the modifier removes only the inset, not the shadow", () => {
  const modifier = rule(".menu-soft-surface.menu-soft-edgeless");
  assert.doesNotMatch(modifier, /inset/);
  assert.match(modifier, /var\(--menu-soft-offset-y\)/);
  assert.match(modifier, /var\(--menu-soft-blur\)/);
  assert.match(modifier, /var\(--menu-soft-spread\)/);
  assert.match(modifier, /var\(--menu-soft-shadow\)/);
});

// Uses vars so the .dark .menu-soft-surface override still applies.
test("the modifier hardcodes neither theme", () => {
  const modifier = rule(".menu-soft-surface.menu-soft-edgeless");
  assert.doesNotMatch(modifier, /rgba|#[0-9a-f]{3}/i);
});

test("the shared vars still match the update banner's shadow", () => {
  assert.match(BANNER, /shadow-\[0_2px_8px_-2px_rgba\(0,0,0,0\.16\)\]/);
  assert.match(BANNER, /dark:shadow-\[0_8px_28px_-6px_var\(--background\)\]/);
  const light = rule(".menu-soft-surface,");
  assert.match(light, /--menu-soft-shadow: rgba\(0, 0, 0, 0\.16\)/);
  assert.match(light, /--menu-soft-offset-y: 2px/);
  assert.match(light, /--menu-soft-blur: 8px/);
  assert.match(light, /--menu-soft-spread: -2px/);
  const dark = rule(".dark .menu-soft-surface,");
  assert.match(dark, /--menu-soft-shadow: var\(--background\)/);
  assert.match(dark, /--menu-soft-offset-y: 8px/);
  assert.match(dark, /--menu-soft-blur: 28px/);
  assert.match(dark, /--menu-soft-spread: -6px/);
});

test("popover and card resolve to the same colour in every theme", () => {
  // Both derive from -base values through the contrast slider (index.css).
  const popover = [...CSS.matchAll(/^\t*--popover-base:\s*([^;]+);/gm)].map(
    (m) => m[1].trim(),
  );
  const card = [...CSS.matchAll(/^\t*--card-base:\s*([^;]+);/gm)].map((m) =>
    m[1].trim(),
  );
  assert.ok(popover.length > 0);
  assert.deepEqual(popover, card);
});
