// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  floatingPanelZIndex,
  useFloatingPanelOrderStore,
} from "../src/lib/floating-panel-order.ts";
import { Z_LAYER } from "../src/lib/z-layers.ts";

test("before either panel is touched neither is singled out", () => {
  assert.equal(
    floatingPanelZIndex("api-monitor", null),
    Z_LAYER.FLOATING_PANEL,
  );
  assert.equal(
    floatingPanelZIndex("resource-monitor", null),
    Z_LAYER.FLOATING_PANEL,
  );
});

test("the panel in front outranks the other one", () => {
  assert.equal(
    floatingPanelZIndex("api-monitor", "api-monitor"),
    Z_LAYER.FLOATING_PANEL_TOP,
  );
  assert.equal(
    floatingPanelZIndex("resource-monitor", "api-monitor"),
    Z_LAYER.FLOATING_PANEL,
  );
});

test("the front panel stays under the layer above the floating panels", () => {
  assert.ok(Z_LAYER.FLOATING_PANEL_TOP < Z_LAYER.STARTUP_SCREEN);
  assert.ok(Z_LAYER.FLOATING_PANEL_TOP < Z_LAYER.TOOLTIP);
});

test("raising swaps which panel is in front", () => {
  const { raise } = useFloatingPanelOrderStore.getState();
  raise("resource-monitor");
  assert.equal(useFloatingPanelOrderStore.getState().top, "resource-monitor");
  raise("api-monitor");
  assert.equal(useFloatingPanelOrderStore.getState().top, "api-monitor");
});

// Runs on pointerdown, so re-raising the front panel must not notify subscribers.
test("raising the panel already in front changes nothing", () => {
  const { raise } = useFloatingPanelOrderStore.getState();
  raise("api-monitor");
  let notified = 0;
  const unsubscribe = useFloatingPanelOrderStore.subscribe(() => {
    notified += 1;
  });
  raise("api-monitor");
  unsubscribe();
  assert.equal(notified, 0);
});

test("a completely hidden panel comes forward whatever is in front", () => {
  assert.equal(
    floatingPanelZIndex("api-monitor", "resource-monitor", true),
    Z_LAYER.FLOATING_PANEL_TOP,
  );
});

test("a panel that is merely behind stays behind", () => {
  assert.equal(
    floatingPanelZIndex("api-monitor", "resource-monitor", false),
    Z_LAYER.FLOATING_PANEL,
  );
});
