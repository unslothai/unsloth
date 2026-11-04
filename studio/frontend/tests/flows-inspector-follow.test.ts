// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useUiStore } = await import("../src/features/flows/stores/ui-store.ts");
const ui = () => useUiStore.getState();

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

test("double-click, I and the toolbar's Settings button open the panel", () => {
  const read = (f: string) =>
    readFileSync(new URL(`../src/features/flows/editor/${f}`, import.meta.url), "utf8");
  assert.match(read("canvas.tsx"), /onNodeDoubleClick[\s\S]*?openInspector\(node\.id\)/);
  assert.match(read("use-editor-shortcuts.ts"), /key === "i"[\s\S]*?toggleInspector\(only\)/);
  assert.match(read("node-toolbar.tsx"), /data-testid="node-toolbar-settings"[\s\S]*?toggleInspector\(nodeId\)/);
});
