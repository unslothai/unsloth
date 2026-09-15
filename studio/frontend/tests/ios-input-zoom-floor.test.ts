// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const css = readSrc("index.css");

// The rule that stops iOS Safari zooming into a focused field: a 16px floor for
// every text-entry control on a coarse pointer. Anything below 16px zooms, and the
// zoom does not come back out, which pushes the header and send button off screen.
const FLOOR_BLOCK =
  /@layer base \{\s*@media \(hover: none\) and \(pointer: coarse\) \{([\s\S]*?)\n\t\}\n\}/;

test("the focus-zoom floor sits in @layer base so it beats important utilities", () => {
  const block = css.match(FLOOR_BLOCK);
  assert.ok(block, "the coarse-pointer font-size floor is gone from index.css");

  // Important declarations reverse layer order, so base beats the utilities layer.
  // An unlayered copy would lose to text-ui-13! and the field would still zoom.
  assert.match(block[1], /font-size:\s*max\(16px, 1rem \* var\(--ui-font-scale, 1\)\) !important;/);
  assert.match(block[1], /\btextarea\b/);
  assert.match(block[1], /\bselect\b/);
});

test("the floor skips controls that render no text of their own", () => {
  const block = css.match(FLOOR_BLOCK);
  assert.ok(block);
  for (const type of [
    "checkbox",
    "radio",
    "range",
    "color",
    "file",
    "button",
    "submit",
    "reset",
    "image",
  ]) {
    assert.match(block[1], new RegExp(`\\[type="${type}"\\]`));
  }
});

// Three controls were measured against 9-12px text and pin a width or height that
// the 16px floor overflows: the value is clipped on a phone or tablet. Each one
// carries a pointer-coarse companion, which leaves every mouse layout untouched.
const COARSE_COMPANIONS: [string, RegExp][] = [
  ["features/studio/sections/progress-section.tsx", /pointer-coarse:h-auto[\s\S]{0,80}pointer-coarse:min-w-0/],
  ["features/studio/sections/params-section-controls.tsx", /w-12 pointer-coarse:w-16/],
  ["features/studio/sections/params-section.tsx", /w-14 pointer-coarse:w-20/],
];

test("the compact controls the floor outgrows keep their coarse-pointer sizing", () => {
  for (const [path, pattern] of COARSE_COMPANIONS) {
    assert.match(readSrc(path), pattern, `${path} lost its coarse-pointer sizing`);
  }
});
