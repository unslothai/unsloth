// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const data = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
const flushes: (() => void)[] = [];
const writes: string[] = [];
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => {
    writes.push(key);
    data.set(key, value);
  },
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: {
    localStorage,
    location: { protocol: "http:" },
    addEventListener: (type: string, flush: () => void) => type === "pagehide" && flushes.push(flush),
  },
});

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserHistoryStore } = await import("../src/features/browser/history-store.ts");
const { useBrowserBookmarksStore } = await import("../src/features/browser/bookmarks-store.ts");
const flush = () => {
  for (const run of flushes) run();
};

test("history actions that change nothing do not rewrite stored history", () => {
  const history = useBrowserHistoryStore.getState();
  history.recordVisit("https://example.com/", "Example");
  history.recordIcon("example.com", "https://example.com/icon.png");
  flush();
  const stored = data.get("unsloth_browser_history:b1");
  assert.match(stored ?? "", /icon\.png/);
  writes.length = 0;
  history.recordIcon("example.com", "https://example.com/icon.png");
  history.recordIcon("", "https://example.com/icon.png");
  history.pruneHistory();
  flush();
  assert.deepEqual(writes, []);
  assert.equal(data.get("unsloth_browser_history:b1"), stored);
});

test("bookmark actions that change nothing do not rewrite stored bookmarks", () => {
  const bookmarks = useBrowserBookmarksStore.getState();
  const bookmark = bookmarks.addBookmark("https://example.com/", "Example");
  assert.ok(bookmark);
  bookmarks.setBookmarkIcon(bookmark.id, "data:image/png;base64,AAAA");
  const stored = data.get("unsloth_browser_bookmarks:b1");
  assert.match(stored ?? "", /AAAA/);
  writes.length = 0;
  bookmarks.setBookmarkIcon(bookmark.id, "data:image/png;base64,AAAA");
  bookmarks.setBookmarkIcon(bookmark.id, "https://example.com/favicon.ico");
  bookmarks.setBookmarkIcon("missing", "data:image/png;base64,BBBB");
  bookmarks.restoreBookmark(bookmark, 0);
  assert.deepEqual(writes, []);
  assert.equal(data.get("unsloth_browser_bookmarks:b1"), stored);
});
