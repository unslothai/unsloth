// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Editor panels in the UI store: the run drawer remembers how the user left it; the settings panel follows selection.

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store: stored } = installLocalStorageFake();

const { useUiStore } = await import("../src/features/flows/stores/ui-store.ts");
const ui = () => useUiStore.getState();

test("the run drawer starts closed and remembers the user's choice", () => {
  assert.equal(ui().drawerOpen, false);
  ui().setDrawer(true);
  assert.equal(stored.get("unsloth_flows_runs_open"), "true");
  ui().setDrawer(false);
  assert.equal(stored.get("unsloth_flows_runs_open"), "false");
});

test("selecting a block leaves a closed settings panel closed", () => {
  ui().reset();
  ui().setSelection(["a"], []);
  assert.equal(ui().inspectorNodeId, null);
});

test("an open panel follows a single selection and holds on multi or empty", () => {
  ui().reset();
  ui().setSelection(["a"], []);
  ui().toggleInspector("a");
  assert.equal(ui().inspectorNodeId, "a");
  ui().setSelection(["b"], []);
  assert.equal(ui().inspectorNodeId, "b");
  ui().setSelection(["b", "c"], []);
  assert.equal(ui().inspectorNodeId, "b");
  ui().setSelection([], []);
  assert.equal(ui().inspectorNodeId, "b");
});

test("toggling the block the panel shows closes it, and it stays closed", () => {
  ui().reset();
  ui().toggleInspector("a");
  ui().toggleInspector("a");
  assert.equal(ui().inspectorNodeId, null);
  ui().setSelection(["c"], []);
  assert.equal(ui().inspectorNodeId, null);
});
