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

test("Settings blurs on a pseudo-element layer; dialogs that scroll their own box keep the direct blur", () => {
  const dialog = ruleAt('html[data-glass][data-glass-menus] [data-slot="dialog-content"] {');
  assert.match(dialog, /\n\s*backdrop-filter: blur\(max\(var\(--glass-blur\), 24px\)\) saturate\(var\(--glass-saturate\)\);/,
    "an absolute layer would scroll away inside dialogs whose own box scrolls (overflow-y-auto)");
  const settings = ruleAt('html[data-glass][data-glass-menus]:not([data-glass-layers]) [data-slot="dialog-content"][data-settings-dialog] {');
  assert.match(settings, /\n\s*backdrop-filter: none;/, "a backdrop-filter on the dialog re-runs for every scrolled frame");
  assert.match(settings, /isolation: isolate;/, "keeps the z-index -1 layer inside the dialog");
  const layer = ruleAt('html[data-glass][data-glass-menus]:not([data-glass-layers]) [data-slot="dialog-content"][data-settings-dialog]::before {');
  assert.match(layer, /z-index: -1;/);
  assert.doesNotMatch(layer, /will-change/, "a promoted layer made the Style dropdown inside Settings slower to open");
  assert.match(layer, /pointer-events: none;/);
  assert.match(layer, /inset: 0;/);
  assert.match(layer, /\n\s*backdrop-filter: blur\(max\(var\(--glass-blur\), 24px\)\) saturate\(var\(--glass-saturate\)\);/);
});

test("the Settings shell never scrolls, so its blur layer stays put", () => {
  const settings = readSrc("features/settings/settings-dialog.tsx");
  assert.match(settings, /data-settings-dialog/);
  assert.match(settings, /"settings-surface [^"]*overflow-hidden"/);
});
