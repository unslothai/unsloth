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

test("light tooltips are the popover surface with a shadow, dark ones flat black", () => {
  const root = css.slice(css.indexOf(":root {"), css.indexOf("}", css.indexOf(":root {")));
  const dark = css.slice(css.indexOf(".dark {"), css.indexOf("}", css.indexOf(".dark {")));
  assert.match(root, /--tooltip-bg: var\(--popover\);/);
  assert.match(root, /--tooltip-shadow: /);
  assert.match(dark, /--tooltip-bg: #000000;/);
  assert.match(dark, /--tooltip-muted: var\(--muted-foreground\);/);
  assert.match(dark, /--tooltip-shadow: none;/);
  assert.match(dark, /--tooltip-shadow-rich: none;/);
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

test("tooltip secondary text is chat's thinking and tool text colour", () => {
  const root = css.slice(css.indexOf(":root {"), css.indexOf("}", css.indexOf(":root {")));
  const trace = /--color-trace-foreground: (color-mix\([^;]+\));/.exec(css)?.[1];
  assert.ok(trace, "--color-trace-foreground is gone");
  assert.ok(root.includes(`--tooltip-muted: ${trace};`), "tooltip secondary text drifted from chat's trace text");
  assert.match(root, /--tooltip-fg: var\(--foreground\);/);
});

test("chart tooltips and native titles use the tooltip surface", () => {
  assert.match(readSrc("components/ui/chart.tsx"), /"tooltip-compact /);
  assert.match(readSrc("features/profile/components/stats/token-activity-card.tsx"), /className="tooltip-compact"/);
  assert.match(readSrc("app/provider.tsx"), /<TitleTooltipLayer \/>/);
  assert.match(readSrc("components/ui/title-tooltip.tsx"), /className="tooltip-compact /);
});

test("every tooltip surface outside TooltipContent rounds to a pill on one line", () => {
  assert.match(readSrc("components/ui/tooltip.tsx"), /roundIfSingleLine\(el\)/);
  assert.match(readSrc("components/ui/title-tooltip.tsx"), /roundIfSingleLine\(tip\)/);
  assert.match(readSrc("components/ui/chart.tsx"), /ref=\{roundIfSingleLine\}/);
  assert.match(readSrc("features/profile/components/stats/token-activity-card.tsx"), /ref=\{roundIfSingleLine\}/);
});
