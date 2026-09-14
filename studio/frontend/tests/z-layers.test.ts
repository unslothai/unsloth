// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The order the named layers are meant to be in. Renumbering is fine; the
// ordering is the contract, and it is the ordering that decides whether a
// Close button can be clicked. tests/studio/test_overlay_layering.py checks the
// components actually use these.

import assert from "node:assert/strict";
import test from "node:test";

import { Z_LAYER } from "../src/lib/z-layers.ts";

// The layers, bottom to top. Adding one belongs in this list.
const ORDER = [
  "OVERLAY_STACK",
  "WINDOW_RESIZE_EDGE",
  "FLOATING_PANEL",
  "FLOATING_PANEL_TOP",
  "STARTUP_SCREEN",
  "TOOLTIP",
  "DRAG_CURSOR_OVERLAY",
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

// What #8199 fixed: a passive status card was covering a window's Close button.
test("floating panels paint over the notification stack", () => {
  assert.ok(Z_LAYER.FLOATING_PANEL > Z_LAYER.OVERLAY_STACK);
});

// The stack reaches the window's bottom edge once it reserves a gutter there,
// and is pointer-active whenever it scrolls. A status card must not be what a
// drag on the window's own grip lands on.
test("the window's bottom resize grips outrank the notification stack", () => {
  assert.ok(Z_LAYER.WINDOW_RESIZE_EDGE > Z_LAYER.OVERLAY_STACK);
  // And stay under a panel being dragged, as they were before.
  assert.ok(Z_LAYER.WINDOW_RESIZE_EDGE < Z_LAYER.FLOATING_PANEL);
});

// Every in-page surface -- dialogs, sheets, dropdowns, the Tauri titlebar --
// is on Tailwind's own scale and tops out at 120. Nothing named here may drop
// into that band, or the numbers stop being comparable at a glance.
test("the named layers stay clear of the in-page scale", () => {
  for (const name of ORDER) {
    assert.ok(
      Z_LAYER[name] > 120,
      `${name} (${Z_LAYER[name]}) has dropped into the in-page z-index band`,
    );
  }
});
