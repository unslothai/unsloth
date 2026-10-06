// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const DAY_MS = 24 * 60 * 60 * 1000;
const data = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
let writes = 0;
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => {
    writes++;
    data.set(key, value);
  },
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: { localStorage, location: { protocol: "http:" }, addEventListener: () => {} },
});

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserHistoryStore } = await import("../src/features/browser/history-store.ts");
const { useBrowserPrefsStore } = await import("../src/features/browser/prefs-store.ts");
const { useBrowserBookmarksStore } = await import("../src/features/browser/bookmarks-store.ts");

const download = { name: "file.pdf", url: "https://example.com/file.pdf", size: 10, contentType: "application/pdf" };

test("with history saving off, visits are not recorded; turned back on, they are", () => {
  const history = useBrowserHistoryStore.getState();
  history.clearHistory();
  useBrowserPrefsStore.getState().setSaveHistory(false);
  history.recordVisit("https://example.com/", "Example");
  assert.equal(useBrowserHistoryStore.getState().history.length, 0);
  useBrowserPrefsStore.getState().setSaveHistory(true);
  history.recordVisit("https://example.com/", "Example");
  assert.equal(useBrowserHistoryStore.getState().history.length, 1);
});

test("with download history off, downloads are not listed", () => {
  const history = useBrowserHistoryStore.getState();
  history.clearDownloads();
  useBrowserPrefsStore.getState().setSaveDownloadHistory(false);
  history.recordDownload(download);
  assert.equal(useBrowserHistoryStore.getState().downloads.length, 0);
  useBrowserPrefsStore.getState().setSaveDownloadHistory(true);
  history.recordDownload(download);
  assert.equal(useBrowserHistoryStore.getState().downloads.length, 1);
});

test("shortening how long history is kept drops older visits at once", () => {
  const now = Date.now();
  useBrowserPrefsStore.getState().setHistoryRetentionDays(0);
  useBrowserHistoryStore.setState({
    history: [
      { id: "new", url: "https://example.com/new", title: "New", visitedAt: now - DAY_MS },
      { id: "old", url: "https://example.com/old", title: "Old", visitedAt: now - 40 * DAY_MS },
    ],
  });
  useBrowserHistoryStore.getState().pruneHistory();
  assert.equal(useBrowserHistoryStore.getState().history.length, 2, "kept until cleared");
  useBrowserPrefsStore.getState().setHistoryRetentionDays(30);
  assert.deepEqual(
    useBrowserHistoryStore.getState().history.map((visit) => visit.id),
    ["new"],
  );
  useBrowserPrefsStore.getState().setHistoryRetentionDays(0);
});

test("an import adds new addresses in one write and skips ones already saved", () => {
  useBrowserBookmarksStore.setState({ bookmarks: [] });
  const store = useBrowserBookmarksStore.getState();
  store.addBookmark("https://example.com/", "Example", "toolbar");
  writes = 0;
  const added = store.importBookmarks([
    { url: "https://example.com/", title: "Duplicate", folder: "other" },
    { url: "https://example.org/", title: "  Org  ", folder: "toolbar", addedAt: 1000 },
    { url: "https://example.net/", title: "Net", folder: "other" },
    { url: "https://example.net/", title: "Net again", folder: "other" },
  ]);
  assert.equal(added, 2);
  assert.equal(writes, 1);
  const bookmarks = useBrowserBookmarksStore.getState().bookmarks;
  assert.deepEqual(
    bookmarks.map((bookmark) => [bookmark.url, bookmark.title, bookmark.folder]),
    [
      ["https://example.com/", "Example", "toolbar"],
      ["https://example.org/", "Org", "toolbar"],
      ["https://example.net/", "Net", "other"],
    ],
  );
  assert.equal(bookmarks[1]?.addedAt, 1000);
  assert.equal(store.importBookmarks([{ url: "https://example.org/", title: "Org", folder: "other" }]), 0);
});
