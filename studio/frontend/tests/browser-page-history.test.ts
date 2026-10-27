// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { parseFrameMessage } = await import("../src/features/browser/frame-message.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");

test("a page's own Back and Forward reach the panel as two flags", () => {
  const from = (message: Record<string, unknown>) => parseFrameMessage({ source: "unsloth-browser", ...message });
  assert.deepEqual(from({ type: "history", back: true, forward: false, url: "https://evil.example/" }), {
    type: "history",
    back: true,
    forward: false,
  });
  assert.deepEqual(from({ type: "history", back: "yes", forward: 1 }), { type: "history", back: false, forward: false });
});

test("a page's Back or Forward that the panel cannot follow leaves its links working", () => {
  // Back on a tab's first page (or Forward at its last) moves nothing, so the frame has not left.
  const source = readSrc("features/browser/page-frame.tsx");
  const latch = source.slice(source.indexOf('if (message.type === "loaded") {'), source.indexOf("onMessageRef.current(message);"));
  assert.ok(latch.includes('(message.type === "navigate" && !message.newTab) || message.type === "reload"'));
  assert.ok(latch.includes('if (left || !tab?.history[tab.index + (message.type === "back" ? -1 : 1)]) return;'));
  assert.equal(latch.split("left = true;").length, 3);
});

test("a page's in-page Back steps are not carried to another page", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://en.wikipedia.org/wiki/Alan_Turing", { newTab: true });
  const id = useBrowserStore.getState().activeTabId ?? "";
  const tab = () => useBrowserStore.getState().tabs.find((candidate) => candidate.id === id);
  store.updateTab(id, { pageHistory: { back: true, forward: false } });
  assert.deepEqual(tab()?.pageHistory, { back: true, forward: false });
  store.navigate(id, { url: "https://en.wikipedia.org/wiki/Albert_Einstein" });
  assert.equal(tab()?.pageHistory, null);
  store.updateTab(id, { pageHistory: { back: true, forward: false } });
  store.goBack(id);
  assert.equal(tab()?.pageHistory, null);
  store.updateTab(id, { pageHistory: { back: false, forward: true } });
  store.reload(id);
  assert.equal(tab()?.pageHistory, null);
});
