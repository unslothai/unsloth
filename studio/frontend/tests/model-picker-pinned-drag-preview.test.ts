// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// Pinned picker rows lift through the sidebar's helpers so the two never drift.

const HOOK = readSrc("features/model-picker/components/model-selector/use-pinned-row-drag.ts");
const SIDEBAR = readSrc("features/chat/hooks/use-sidebar-drag.ts");
const PICKERS = readSrc("features/model-picker/components/model-selector/pickers.tsx");
const CSS = readSrc("index.css");

test("the sidebar's lift, follow and cue helpers are shared", () => {
  for (const name of ["liftCopy", "placeGhost", "placeCue"]) {
    assert.match(SIDEBAR, new RegExp(`export function ${name}\\(`), name);
  }
  assert.match(SIDEBAR, /liftCopy\(face, pressY, view, ROW_GHOST_CLASS, GHOST_DROPPED_ATTRS\)/);
});

test("a carried Pinned row lifts a copy that follows the pointer", () => {
  assert.match(
    HOOK,
    /started = true;\n\s*scroller\.current = scrollerOf\(row\);\n\s*ghost\.current = liftCopy\(\n\s*faceOf\(row\),/,
  );
  assert.match(HOOK, /if \(ghost\.current\) placeGhost\(ghost\.current, at\.y\);/);
  for (const attr of ['"id"', "SCOPE_ATTR", "KEY_ATTR", '"data-model-picker-option"', '"data-state"']) {
    assert.match(HOOK, new RegExp(`GHOST_DROPPED_ATTRS = \\[[^\\]]*${attr.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`), attr);
  }
  assert.match(CSS, /\.model-picker-row-ghost \{\n[^}]*position: fixed;[^}]*z-index: 60;/);
  assert.match(CSS, /\.model-picker-row-ghost \* \{\n\tpointer-events: none;/);
});

test("every copy is a faintly frosted shade darker than the list it came from", () => {
  const ghosts = String.raw`:is\(\.sidebar-row-ghost, \.sidebar-section-ghost, \.model-picker-row-ghost\)`;
  const light = new RegExp(String.raw`\n${ghosts} \{([^}]*)\}`).exec(CSS);
  const dark = new RegExp(String.raw`\.dark ${ghosts} \{([^}]*)\}`).exec(CSS);
  assert.ok(light, "shared ghost look");
  assert.ok(dark, "shared ghost look in dark mode");
  assert.match(light[1], /background: color-mix\(in oklab, color-mix\(in oklab, var\(--row-ghost-surface\), black 4%\) 62%, transparent\);/);
  assert.match(dark[1], /background: color-mix\(in oklab, color-mix\(in oklab, var\(--row-ghost-surface\), black 22%\) 62%, transparent\);/);
  assert.match(light[1], /\bbackdrop-filter: blur\(0\.5px\) saturate\(1\.05\);/);
  assert.match(light[1], /0 0 0 1px rgb\(0 0 0 \/ 0\.06\)/);
  assert.match(dark[1], /0 0 0 1px rgb\(255 255 255 \/ 0\.045\)/);
  assert.match(CSS, /\.sidebar-row-ghost \{\n\t--row-ghost-surface: var\(--sidebar\);/);
  assert.match(CSS, /\.sidebar-section-ghost \{\n\t--row-ghost-surface: var\(--sidebar\);/);
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
  assert.match(CSS, /\.model-picker-row-ghost > \* \{\n[^}]*margin: 0;/);
  assert.match(HOOK, /const ROW_FACE_ATTR = "data-pinned-row-face";/);
  assert.match(HOOK, /const faceOf = \(row: Element\): HTMLElement =>\n\s*row\.querySelector<HTMLElement>\(`\[\$\{ROW_FACE_ATTR\}\]`\) \?\?/);
  assert.match(
    PICKERS,
    /<div key=\{adapter\.id\}>\n\s*<div\n\s*className=\{downloadedRowShellClassName\(value === adapter\.id\)\}\n[^\n]*\n\s*data-pinned-row-face=""/,
  );
});
