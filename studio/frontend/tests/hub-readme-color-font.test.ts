// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { stripColorFontTriggers } from "../src/features/hub/lib/color-font-triggers.ts";

// Issue #9453: a Linux AppImage's bundled WebKitGTK/Skia asserts on a COLRv1
// color-stop table when a README renders a color-font glyph.
test("strips a plain pictograph emoji", () => {
  assert.equal(
    stripColorFontTriggers("# Qwen3.8-27B \u{1F680} Fast and accurate!"),
    "# Qwen3.8-27B  Fast and accurate!",
  );
});

test("strips a regional-indicator flag pair", () => {
  assert.equal(
    stripColorFontTriggers("Supports \u{1F1EC}\u{1F1E7} and \u{1F1FA}\u{1F1F8} locales."),
    "Supports  and  locales.",
  );
});

test("strips a ZWJ-joined family emoji sequence entirely", () => {
  assert.equal(
    stripColorFontTriggers(
      "Family: \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}",
    ),
    "Family: ",
  );
});

test("strips a variation-selector-16 star and checkmark", () => {
  assert.equal(
    stripColorFontTriggers("Star rating ⭐ and checkmark ✅."),
    "Star rating  and checkmark .",
  );
});

test("leaves plain markdown, code, links, and arrows untouched", () => {
  const markdown =
    "Plain markdown: **bold**, `code`, [link](https://x.com), 100% -> arrow, a->b";
  assert.equal(stripColorFontTriggers(markdown), markdown);
});
