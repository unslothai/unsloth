// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");

function ruleAt(head: string): string {
  const at = CSS.indexOf(head);
  assert.ok(at >= 0, `missing rule ${head}`);
  return CSS.slice(at, CSS.indexOf("}", at) + 1);
}

test("the glass dialog blurs on a pseudo-element layer, not on itself", () => {
  const dialog = ruleAt('html[data-glass][data-glass-menus] [data-slot="dialog-content"] {');
  assert.doesNotMatch(dialog, /backdrop-filter/, "a backdrop-filter on the dialog re-runs for every scrolled frame");
  assert.match(dialog, /isolation: isolate;/, "keeps the z-index -1 layer inside the dialog");
  const layer = ruleAt('html[data-glass][data-glass-menus] [data-slot="dialog-content"]::before {');
  assert.match(layer, /z-index: -1;/);
  assert.match(layer, /will-change: transform;/);
  assert.match(layer, /pointer-events: none;/);
  assert.match(layer, /inset: 0;/);
  assert.match(layer, /\n\s*backdrop-filter: blur\(max\(var\(--glass-blur\), 24px\)\) saturate\(var\(--glass-saturate\)\);/);
});
