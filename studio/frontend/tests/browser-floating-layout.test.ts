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
  assert.equal(floatingNativeZoom(0.25, 1), 0.25);
});

test("only a page shown natively skips the floating window's scale; files and new tabs scale even with native browsing on", async () => {
  const { readFileSync } = await import("node:fs");
  const panel = readFileSync(new URL("../src/features/browser/browser-panel.tsx", import.meta.url), "utf8");
  assert.match(panel, /const nativeShown = native && activeEntry\?\.kind === "web" && !activeTab\?\.nativeError;/);
  assert.match(panel, /floating && !nativeShown\s*\?\s*FLOATING_PAGE_STYLE/);
});

test("floating is the chat's own: another chat opens its browser beside it, and the first floats again on return", () => {
  const store = () => useBrowserStore.getState();
  store().showChat("single:float-a");
  store().openUrl("https://example.com/a");
  store().setFloating(true);
  store().showChat("single:float-b");
  store().openUrl("https://example.com/b");
  assert.equal(store().floating, false);
  store().showChat("single:float-a");
  assert.deepEqual([store().open, store().floating], [true, true]);
});

test("a browser floated by a close mid-run is still that on return from another chat, and its close restores the layout", () => {
  const store = () => useBrowserStore.getState();
  store().showChat("single:agent-a");
  store().openUrl("https://example.com/a");
  store().floatForAgent();
  store().showChat("single:agent-b");
  assert.deepEqual([store().floating, store().floatedForAgent], [false, false]);
  store().showChat("single:agent-a");
  assert.deepEqual([store().open, store().floating, store().floatedForAgent], [true, true, true]);
  store().closePanel();
  assert.deepEqual([store().open, store().floating, store().floatedForAgent], [false, false, false]);
});

test("the agent working in the floated browser while another chat shows keeps it floated for the agent", () => {
  const store = () => useBrowserStore.getState();
  store().showChat("single:agent-c");
  store().openUrl("https://example.com/c");
  const tab = store().activeTabId ?? "";
  store().floatForAgent();
  store().showChat("single:agent-d");
  store().revealIn("single:agent-c", tab);
  store().openTabIn("single:agent-c", "https://example.com/c2", { background: true });
  store().showChat("single:agent-c");
  assert.deepEqual([store().floating, store().floatedForAgent], [true, true]);
  store().closePanel();
  assert.deepEqual([store().open, store().floating], [false, false]);
});
