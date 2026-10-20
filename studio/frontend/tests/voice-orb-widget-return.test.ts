// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { onMainFocus, returnFromWidget } from "../src/features/chat/voice/orb-widget-return.ts";

function rig(floating: boolean) {
  const calls: string[] = [];
  const w = { floating };
  const setCollapsed = (c: boolean) => calls.push(`collapsed:${c}`);
  const showMain = () => calls.push("showMain");
  return { w, calls, setCollapsed, showMain };
}

test("back from the widget shows the in-app orb again and hands the window back", () => {
  const r = rig(true);
  assert.equal(returnFromWidget(r.w, r.setCollapsed, r.showMain), true);
  assert.equal(r.w.floating, false);
  assert.deepEqual(r.calls, ["collapsed:false", "showMain"]);
  // A second return (the Back button after a Dock click) does nothing.
  assert.equal(returnFromWidget(r.w, r.setCollapsed, r.showMain), false);
  assert.deepEqual(r.calls, ["collapsed:false", "showMain"]);
});

test("the main window taking focus while floating (Dock, tray, menu) is a return", () => {
  const r = rig(true);
  assert.equal(onMainFocus(false, r.w, r.setCollapsed, r.showMain), false);
  assert.equal(r.w.floating, true);
  assert.equal(onMainFocus(true, r.w, r.setCollapsed, r.showMain), true);
  assert.deepEqual(r.calls, ["collapsed:false", "showMain"]);
});

test("focus with no widget up leaves the orb as the user set it", () => {
  const r = rig(false);
  assert.equal(onMainFocus(true, r.w, r.setCollapsed, r.showMain), false);
  assert.deepEqual(r.calls, []);
});
