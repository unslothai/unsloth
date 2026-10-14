// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The permission shield sits right after the composer plus. While the plus spun open or closed,
// WebKit gave the shield's positioned slot its own layer and snapped it a device pixel up, then
// dropped the layer and let it fall back. Stacking the icon and X in a grid cell keeps the slot
// unpositioned, so it always paints with the pill.

const CSS = readSrc("index.css");

function rule(selector: string): string {
  const match = new RegExp(`\\n\\t${selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} \\{([^}]*)\\}`).exec(CSS);
  assert.ok(match, `missing ${selector}`);
  return match[1];
}

test("the pill glyph slot is an unpositioned grid", () => {
  const glyph = rule(".composer-pill-glyph");
  assert.match(glyph, /@apply inline-grid w-\[var\(--ui-icon-size\)\] shrink-0 place-items-center/);
  assert.doesNotMatch(glyph, /\brelative\b|\babsolute\b/);
  assert.match(rule(".composer-pill-glyph > *"), /grid-area: 1 \/ 1;/);
});

test("the hover X shares the icon's cell instead of overlaying it", () => {
  const x = rule(".composer-pill-x");
  assert.match(x, /@apply pointer-events-none size-\[var\(--ui-icon-size\)\] rounded-full/);
  assert.doesNotMatch(x, /\babsolute\b|\binset-0\b|\bm-auto\b/);
});
