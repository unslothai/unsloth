// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const data = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: { localStorage, location: { protocol: "http:" }, addEventListener: () => {} },
});

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserBookmarksStore } = await import("../src/features/browser/bookmarks-store.ts");

test("bookmarks persist under the account that saved them", () => {
  useBrowserBookmarksStore.getState().addBookmark("https://example.com/", "Example");
  assert.equal(data.has("unsloth_browser_bookmarks"), false);
  assert.match(data.get("unsloth_browser_bookmarks:b1") ?? "", /example\.com/);
});

test("a page is bookmarked once, and the next one goes where the last was moved", () => {
  const store = useBrowserBookmarksStore.getState();
  const first = store.addBookmark("https://example.com/", "Again");
  assert.equal(first?.title, "Example");
  assert.equal(useBrowserBookmarksStore.getState().bookmarks.length, 1);
  store.updateBookmark(first?.id ?? "", { title: "  Renamed  ", folder: "other" });
  const second = store.addBookmark("https://example.org/", "Org");
  assert.equal(second?.folder, "other");
  assert.equal(useBrowserBookmarksStore.getState().bookmarks[0]?.title, "Renamed");
});

test("an address too long to keep is not saved, and a removed bookmark can be put back in place", () => {
  const store = useBrowserBookmarksStore.getState();
  assert.equal(store.addBookmark(`https://example.com/${"q".repeat(5000)}`, "Long"), null);
  const [first] = useBrowserBookmarksStore.getState().bookmarks;
  assert.ok(first);
  store.removeBookmark(first.id);
  assert.equal(useBrowserBookmarksStore.getState().bookmarks.length, 1);
  store.restoreBookmark(first, 0);
  assert.deepEqual(
    useBrowserBookmarksStore.getState().bookmarks.map((bookmark) => bookmark.url),
    ["https://example.com/", "https://example.org/"],
  );
});

test("a bookmark keeps a small image icon, and nothing that isn't one", () => {
  const store = useBrowserBookmarksStore.getState();
  const [bookmark] = useBrowserBookmarksStore.getState().bookmarks;
  assert.ok(bookmark);
  store.setBookmarkIcon(bookmark.id, "https://example.com/favicon.ico");
  store.setBookmarkIcon(bookmark.id, `data:image/png;base64,${"A".repeat(100_000)}`);
  assert.equal(useBrowserBookmarksStore.getState().bookmarks[0]?.icon, undefined);
  store.setBookmarkIcon(bookmark.id, "data:image/png;base64,AAAA");
  assert.equal(useBrowserBookmarksStore.getState().bookmarks[0]?.icon, "data:image/png;base64,AAAA");
});
