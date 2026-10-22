// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { parseFrameMessage } = await import("../src/features/browser/frame-message.ts");
const { currentEntry, saveScroll, savedScroll, useBrowserStore } = await import("../src/features/browser/store.ts");

test("a page's scroll position reaches the panel as numbers only", () => {
  const from = (message: Record<string, unknown>) => parseFrameMessage({ source: "unsloth-browser", ...message });
  assert.deepEqual(from({ type: "scroll", x: 0, y: 4000, extra: 1 }), { type: "scroll", x: 0, y: 4000 });
  assert.equal(from({ type: "scroll", x: 0, y: "4000" }), null);
  assert.equal(from({ type: "scroll", x: 0, y: Number.NaN }), null);
});

test("Back and Forward return each page to where it was scrolled", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://en.wikipedia.org/wiki/Alan_Turing", { newTab: true });
  const id = useBrowserStore.getState().activeTabId ?? "";
  const shown = () => currentEntry(useBrowserStore.getState().tabs.find((tab) => tab.id === id)!);
  saveScroll(shown(), { x: 0, y: 4000 });
  store.navigate(id, { url: "https://en.wikipedia.org/wiki/Albert_Einstein" });
  assert.equal(savedScroll(shown()), undefined);
  saveScroll(shown(), { x: 0, y: 1200 });
  store.goBack(id);
  assert.deepEqual(savedScroll(shown()), { x: 0, y: 4000 });
  store.goForward(id);
  assert.deepEqual(savedScroll(shown()), { x: 0, y: 1200 });
  store.goBack(id);
  store.navigate(id, { url: "https://en.wikipedia.org/wiki/Albert_Einstein" });
  assert.equal(savedScroll(shown()), undefined);
});

test("a scroll taken on a route the page moved to is not applied to the page it was opened at", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://huggingface.co/models", { newTab: true });
  const id = useBrowserStore.getState().activeTabId ?? "";
  const shown = () => currentEntry(useBrowserStore.getState().tabs.find((tab) => tab.id === id)!);
  saveScroll(shown(), { x: 0, y: 900 }, "https://huggingface.co/models?search=qwen");
  assert.equal(savedScroll(shown()), undefined);
  saveScroll(shown(), { x: 0, y: 300 }, "https://huggingface.co/models#top");
  assert.deepEqual(savedScroll(shown()), { x: 0, y: 300 });
});

test("Reload keeps the scroll of a page that moved its route", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://huggingface.co/models", { newTab: true });
  const id = useBrowserStore.getState().activeTabId ?? "";
  const shown = () => currentEntry(useBrowserStore.getState().tabs.find((tab) => tab.id === id)!);
  store.updateTab(id, { displayUrl: "https://huggingface.co/models?search=qwen" });
  saveScroll(shown(), { x: 0, y: 900 }, "https://huggingface.co/models?search=qwen");
  store.reload(id);
  const entry = shown();
  assert.equal(entry.kind === "web" && entry.url, "https://huggingface.co/models?search=qwen");
  assert.deepEqual(savedScroll(shown()), { x: 0, y: 900 });
});

test("a page that redirected gets its scroll back on Back", () => {
  const store = useBrowserStore.getState();
  store.openUrl("http://example.com/docs", { newTab: true });
  const id = useBrowserStore.getState().activeTabId ?? "";
  const shown = () => currentEntry(useBrowserStore.getState().tabs.find((tab) => tab.id === id)!);
  saveScroll(shown(), { x: 0, y: 700 }, "https://example.com/docs/");
  assert.deepEqual(savedScroll(shown(), "https://example.com/docs/"), { x: 0, y: 700 });
  assert.equal(savedScroll(shown(), "https://example.com/docs/other"), undefined);
});
