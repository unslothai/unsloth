// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// Painted on the glyph box, the circle kept subpixel precision while the icon snapped.

const CSS = readSrc("index.css");
const SIDEBAR = readSrc("components/app-sidebar.tsx");

test("the hover circle is painted on the icon, padded out to the glyph box", () => {
  assert.match(
    CSS,
    /\.sidebar-row-action-glyph > svg \{\n\t\t--glyph-icon: var\(--icon-size\);\n\t\tbox-sizing: border-box;\n\t\twidth: 100% !important;\n\t\theight: 100% !important;\n\t\tpadding: calc\(\(100% - var\(--glyph-icon\)\) \/ 2\);\n\t\tborder-radius: 9999px;/,
  );
  assert.match(CSS, /\.sidebar-row-action-glyph > svg\.size-4 \{\n\t\t--glyph-icon: calc\(var\(--spacing\) \* 4\);/);
  assert.match(CSS, /\.sidebar-row-action-glyph > svg\.size-3\\\.5 \{\n\t\t--glyph-icon: calc\(var\(--spacing\) \* 3\.5\);/);
  // The tint is on the icon, not the glyph box, or the two would snap apart again.
  const hover = /\.sidebar-row-action:hover \.sidebar-row-action-glyph,\n\t\.sidebar-row-action\[data-state="open"\] \.sidebar-row-action-glyph \{([^}]*)\}/.exec(CSS);
  assert.ok(hover);
  assert.doesNotMatch(hover[1], /background-color/);
  assert.match(
    CSS,
    /\.sidebar-row-action:hover \.sidebar-row-action-glyph > svg,\n\t\.sidebar-row-action\[data-state="open"\] \.sidebar-row-action-glyph > svg \{\n[^}]*background-color: color-mix/,
  );
});

test("the row kebab draws its dots on the circle's centre", () => {
  // Hugeicons draws them around y = 12.5 of a 24-unit box.
  assert.match(SIDEBAR, /\{ \.\.\.attrs, transform: "translate\(0 -0\.5\)" \}/);
  assert.equal(SIDEBAR.includes("icon={MoreVerticalIcon}"), false);
  assert.equal((SIDEBAR.match(/icon=\{MoreVerticalCenteredIcon\}/g) ?? []).length, 3);
});
