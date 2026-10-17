// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The ordering is the contract, not the numbers. tests/studio/test_overlay_layering.py checks usage.

import assert from "node:assert/strict";
import test from "node:test";

import { Z_LAYER } from "../src/lib/z-layers.ts";

const ORDER = [
  "OVERLAY_STACK",
  "WINDOW_RESIZE_EDGE",
  "FLOATING_PANEL",
  "FLOATING_PANEL_TOP",
  "STARTUP_SCREEN",
  "TOOLTIP",
  "DRAG_CURSOR_OVERLAY",
  "WINDOW_BARS",
  "ZOOM_POPUP",
] as const;

test("the named layers are strictly ordered", () => {
  for (let i = 1; i < ORDER.length; i += 1) {
    const below = ORDER[i - 1];
    const above = ORDER[i];
    assert.ok(
      Z_LAYER[below] < Z_LAYER[above],
      `${below} (${Z_LAYER[below]}) must sit under ${above} (${Z_LAYER[above]})`,
    );
  }
});

test("every layer is listed in the order", () => {
  assert.deepEqual(Object.keys(Z_LAYER).sort(), [...ORDER].sort());
});

test("floating panels paint over the notification stack", () => {
  assert.ok(Z_LAYER.FLOATING_PANEL > Z_LAYER.OVERLAY_STACK);
});

// The stack reaches the bottom edge and is pointer-active while scrolling, so grips must outrank it.
test("the window's bottom resize grips outrank the notification stack", () => {
  assert.ok(Z_LAYER.WINDOW_RESIZE_EDGE > Z_LAYER.OVERLAY_STACK);
  assert.ok(Z_LAYER.WINDOW_RESIZE_EDGE < Z_LAYER.FLOATING_PANEL);
});

// In-page surfaces use Tailwind's scale up to 120; named layers stay clear of that band.
test("the named layers stay clear of the in-page scale", () => {
  for (const name of ORDER) {
    assert.ok(
      Z_LAYER[name] > 120,
      `${name} (${Z_LAYER[name]}) has dropped into the in-page z-index band`,
    );
  }
});
