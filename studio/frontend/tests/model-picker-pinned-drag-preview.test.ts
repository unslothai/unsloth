// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// A Pinned row carried in the model picker lifts as a copy under the pointer, as a sidebar chat
// does, through the same helpers so the two never drift. Every tab (Chat, Images, Audio, Video)
// draws its picker with HubModelPicker, so they all carry rows this way.

const HOOK = readSrc("features/model-picker/components/model-selector/use-pinned-row-drag.ts");
const SIDEBAR = readSrc("features/chat/hooks/use-sidebar-drag.ts");
const PICKERS = readSrc("features/model-picker/components/model-selector/pickers.tsx");
const CSS = readSrc("index.css");

test("the sidebar's lift, follow and cue helpers are shared", () => {
  for (const name of ["liftCopy", "placeGhost", "placeCue"]) {
    assert.match(SIDEBAR, new RegExp(`export function ${name}\\(`), name);
  }
  // The sidebar still lifts its rows through the shared copy.
  assert.match(SIDEBAR, /liftCopy\(face, pressY, view, ROW_GHOST_CLASS, GHOST_DROPPED_ATTRS\)/);
});

test("a carried Pinned row lifts a copy that follows the pointer", () => {
  assert.match(
    HOOK,
    /started = true;\n\s*scroller\.current = scrollerOf\(row\);\n\s*ghost\.current = liftCopy\(\n\s*faceOf\(row\),/,
  );
  assert.match(HOOK, /if \(ghost\.current\) placeGhost\(ghost\.current, at\.y\);/);
  // A picture only: never an option the list's keys, the drop hit test or a tooltip finds.
  for (const attr of ['"id"', "SCOPE_ATTR", "KEY_ATTR", '"data-model-picker-option"', '"data-state"']) {
    assert.match(HOOK, new RegExp(`GHOST_DROPPED_ATTRS = \\[[^\\]]*${attr.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`), attr);
  }
  assert.match(CSS, /\.model-picker-row-ghost \{\n[^}]*position: fixed;[^}]*z-index: 60;/);
  assert.match(CSS, /\.model-picker-row-ghost \* \{\n\tpointer-events: none;/);
});

test("both copies are a translucent shade off the list they came from", () => {
  const look = /:is\(\.sidebar-row-ghost, \.model-picker-row-ghost\) \{([^}]*)\}/.exec(CSS);
  assert.ok(look, "shared ghost look");
  assert.match(look[1], /background: color-mix\(in oklab, color-mix\(in oklab, var\(--row-ghost-surface\), var\(--foreground\) 5%\) 85%, transparent\);/);
  assert.match(look[1], /backdrop-filter: blur\(8px\);/);
  assert.match(CSS, /\.sidebar-row-ghost \{\n\t--row-ghost-surface: var\(--sidebar\);/);
  assert.match(CSS, /\.model-picker-row-ghost \{\n\t--row-ghost-surface: var\(--popover\);/);
});

test("the drop line is drawn again above the copy", () => {
  // Read at render: the chat barrel imports the picker back, so a module-scope read is undefined.
  assert.match(PICKERS, /className=\{cn\("relative", edge && \[DROP_CUE_CLASS, PINNED_DROP_CUE\[edge\]\]\)\}/);
  assert.match(HOOK, /showTarget\(aim\(at\.x, at\.y\)\);\n[^\n]*\n\s*if \(ghost\.current\) placeCue\(ghost\.current\);/);
});

test("every way a drag ends takes the copy down, and a drop settles into place", () => {
  assert.match(
    HOOK,
    /const clear = useCallback\(\(\) => \{[^]*?ghost\.current\?\.element\.remove\(\);\n\s*for \(const overlay of ghost\.current\?\.cues \?\? \[\]\) overlay\.remove\(\);\n\s*ghost\.current = null;/,
  );
  assert.match(HOOK, /if \(from !== null && !prefersReducedMotion\(\)\) \{\n\s*settleRow\(optionsRef\.current\.scope, key, from\);/);
});

test("the copy sits on its own pill for every Pinned row kind", () => {
  // A Connected row's ml-4 is not part of the pill the copy is sized to, so it must not carry over.
  assert.match(CSS, /\.model-picker-row-ghost > \* \{\n[^}]*margin: 0;/);
  // A fine-tuned row nests its pill in a keyed wrapper; the drag lifts the marked pill instead.
  assert.match(HOOK, /const ROW_FACE_ATTR = "data-pinned-row-face";/);
  assert.match(HOOK, /const faceOf = \(row: Element\): HTMLElement =>\n\s*row\.querySelector<HTMLElement>\(`\[\$\{ROW_FACE_ATTR\}\]`\) \?\?/);
  assert.match(
    PICKERS,
    /<div key=\{adapter\.id\}>\n\s*<div\n\s*className=\{downloadedRowShellClassName\(value === adapter\.id\)\}\n[^\n]*\n\s*data-pinned-row-face=""/,
  );
});
