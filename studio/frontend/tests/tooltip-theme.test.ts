// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const css = readSrc("index.css");

function rule(selector: string): string {
  const start = css.indexOf(`${selector} {`);
  assert.ok(start >= 0, `${selector} is gone`);
  return css.slice(start, css.indexOf("}", start));
}

// #13997: tooltips were black in both themes, with theme grey text on them.
test("tooltip surfaces follow the theme instead of a fixed black", () => {
  for (const selector of [".tooltip-compact", ".tooltip-rich"]) {
    const body = rule(selector);
    assert.match(body, /bg-\(--tooltip-bg\)/, selector);
    assert.match(body, /text-\(--tooltip-fg\)/, selector);
    assert.doesNotMatch(body, /bg-black|text-white/, selector);
  }
  for (const pseudo of css.matchAll(/content: attr\(data-pill-label\);\s*@apply ([^;]+);/g)) {
    assert.match(pseudo[1], /bg-\(--tooltip-bg\)/, "a pill label tooltip is black again");
  }
  assert.doesNotMatch(css, /#b1b1b1/, "a tooltip label colour assumes a black surface");
});

test("light tooltips are the popover surface and dark ones a lifted grey", () => {
  const root = css.slice(css.indexOf(":root {"), css.indexOf("}", css.indexOf(":root {")));
  const dark = css.slice(css.indexOf(".dark {"), css.indexOf("}", css.indexOf(".dark {")));
  assert.match(root, /--tooltip-bg: var\(--popover\);/);
  assert.match(root, /--tooltip-shadow: /);
  assert.match(dark, /--tooltip-bg: #444444;/);
});

test("tooltip contents use the tooltip tokens, not white-on-black tints", () => {
  for (const file of [
    "components/assistant-ui/sources.tsx",
    "features/hub/catalog/model-inspector.tsx",
    "features/hub/catalog/models-catalog-rows.tsx",
  ]) {
    assert.doesNotMatch(readSrc(file), /bg-black|text-white\/\d+/, file);
  }
});
