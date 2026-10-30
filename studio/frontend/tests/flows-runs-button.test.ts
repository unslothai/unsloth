// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The run drawer opens only from the Runs button, and remembers how the user left it.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const stored = new Map<string, string>();
(globalThis as { localStorage?: unknown }).localStorage = {
  getItem: (k: string) => stored.get(k) ?? null,
  setItem: (k: string, v: string) => void stored.set(k, v),
  removeItem: (k: string) => void stored.delete(k),
};
const { useUiStore } = await import("../src/features/flows/stores/ui-store.ts");

test("nothing in a run opens the drawer: only the Runs button and the drawer's own tabs call setDrawer", () => {
  assert.doesNotMatch(readSrc("features/flows/editor/run-controller.ts"), /setDrawer/);
  assert.doesNotMatch(readSrc("features/flows/stores/runtime-store.ts"), /setDrawer|drawerOpen/);
  const bar = readSrc("features/flows/editor/run-bar.tsx");
  assert.match(bar, /onClick=\{\(\) => useUiStore\.getState\(\)\.setDrawer\(!open, "run"\)\}/);
  assert.match(bar, /data-testid="flows-runs"/);
  assert.match(bar, /aria-pressed=\{open\}/);
});

test("the drawer starts closed and remembers the user's choice", () => {
  assert.equal(useUiStore.getState().drawerOpen, false);
  useUiStore.getState().setDrawer(true);
  assert.equal(stored.get("unsloth_flows_runs_open"), "true");
  useUiStore.getState().setDrawer(false);
  assert.equal(stored.get("unsloth_flows_runs_open"), "false");
});

test("a finished run pulses the button once in its status colour, a failed one in red", () => {
  const bar = readSrc("features/flows/editor/run-bar.tsx");
  assert.match(bar, /failed: "var\(--destructive\)"/);
  assert.match(bar, /if \(was\.current && !busy && ink\)/);
  assert.match(readSrc("features/flows/editor/flows-editor.css"), /\.flows-runs-pulse \{ animation: flows-runs-pulse/);
});
