// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const css = readSrc("index.css");

// Below 16px iOS Safari zooms into a focused field and never zooms back out.
const FLOOR_BLOCK =
  /@layer base \{\s*@media \(hover: none\) and \(pointer: coarse\) \{([\s\S]*?)\n\t\}\n\}/;

test("the focus-zoom floor sits in @layer base so it beats important utilities", () => {
  const block = css.match(FLOOR_BLOCK);
  assert.ok(block, "the coarse-pointer font-size floor is gone from index.css");

  // Important declarations reverse layer order, so base beats text-ui-13!.
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

// Three controls pin a box sized for 9-12px text, so the floor clips their value.
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
