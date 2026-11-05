// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserStore } = await import("../src/features/browser/store.ts");

test("floating, full view and the split are one layout at a time, and floating outlasts a close", () => {
  const store = () => useBrowserStore.getState();
  store().openUrl("https://example.com");
  store().setFullView(true);
  store().setFloating(true);
  assert.deepEqual([store().floating, store().fullView], [true, false]);
  store().setFullView(true);
  assert.deepEqual([store().floating, store().fullView], [false, true]);
  store().setFloating(true);
  store().splitWithChatOn("right");
  assert.deepEqual([store().floating, store().fullView, store().chatSide], [false, false, "right"]);
  store().setFloating(true);
  store().closePanel();
  store().openPanel();
  assert.deepEqual([store().open, store().floating], [true, true]);
});

test("closed while the model still works in it, the browser floats on the model's tab; closing the preview closes it", async () => {
  const { patchAgentChat } = await import("../src/features/browser/agent/agent-store.ts");
  const { closeOrFloatBrowser } = await import("../src/features/browser/close-browser.ts");
  const store = () => useBrowserStore.getState();
  store().setFloating(false);
  store().openUrl("https://example.com/agent", { newTab: true });
  const agentTab = store().activeTabId;
  store().openUrl("https://example.com/mine", { newTab: true });
  patchAgentChat(store().chat, { tabId: agentTab, runKey: "run-1", stoppedRun: null });
  closeOrFloatBrowser();
  assert.deepEqual([store().open, store().floating, store().activeTabId], [true, true, agentTab]);
  closeOrFloatBrowser();
  // The float was the run's, not a layout the reader chose: the next open is beside the chat again.
  assert.deepEqual([store().open, store().floating], [false, false]);
  store().openPanel();
  patchAgentChat(store().chat, { stoppedRun: "run-1" });
  closeOrFloatBrowser();
  assert.equal(store().open, false);
});

test("a native page zoomed out before floating stays at the native zoom floor instead of failing", async () => {
  const { NATIVE_MIN_ZOOM, floatingNativeZoom } = await import("../src/features/browser/floating-events.ts");
  assert.equal(floatingNativeZoom(1, 0.5), 0.5);
  assert.equal(floatingNativeZoom(0.25, 0.615), NATIVE_MIN_ZOOM);
  assert.equal(floatingNativeZoom(0.4, 0.5), NATIVE_MIN_ZOOM);
  // Docked, the tab's own zoom goes through untouched.
  assert.equal(floatingNativeZoom(0.25, 1), 0.25);
});
