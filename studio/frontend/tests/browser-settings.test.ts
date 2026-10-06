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
  assert.deepEqual(added, { added: 2, leftOut: 0 });
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
  assert.deepEqual(store.importBookmarks([{ url: "https://example.org/", title: "Org", folder: "other" }]), {
    added: 0,
    leftOut: 0,
  });
});

test("reloading the latest page still drops visits past the kept period", () => {
  const now = Date.now();
  // Set first, so the visit below only crosses the cutoff while Studio is open.
  useBrowserPrefsStore.getState().setHistoryRetentionDays(30);
  useBrowserHistoryStore.setState({
    history: [
      { id: "latest", url: "https://example.com/a", title: "A", visitedAt: now - 1000 },
      { id: "old", url: "https://example.com/old", title: "Old", visitedAt: now - 40 * DAY_MS },
    ],
  });
  useBrowserHistoryStore.getState().recordVisit("https://example.com/a", "A");
  assert.deepEqual(
    useBrowserHistoryStore.getState().history.map((visit) => visit.id),
    ["latest"],
  );
  useBrowserPrefsStore.getState().setHistoryRetentionDays(0);
});

test("with history saving off, site icons are not kept either", () => {
  useBrowserHistoryStore.getState().clearHistory();
  useBrowserPrefsStore.getState().setSaveHistory(false);
  useBrowserHistoryStore.getState().recordIcon("example.com", "https://example.com/icon.png");
  assert.deepEqual(useBrowserHistoryStore.getState().icons, {});
  useBrowserPrefsStore.getState().setSaveHistory(true);
  useBrowserHistoryStore.getState().recordIcon("example.com", "https://example.com/icon.png");
  assert.equal(useBrowserHistoryStore.getState().icons["example.com"], "https://example.com/icon.png");
});

test("a fetched file shows at 100% and the next page returns to the default zoom", async () => {
  const { useBrowserStore, setPageDownload } = await import("../src/features/browser/store.ts");
  const { fitZoomToPage, homeZoom } = await import("../src/features/browser/zoom.ts");
  useBrowserPrefsStore.getState().setDefaultZoom(1.25);
  useBrowserStore.getState().openUrl("https://example.com/paper.pdf", { newTab: true });
  const tabId = useBrowserStore.getState().activeTabId ?? "";
  const zoom = () => useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.zoom;
  assert.equal(zoom(), 1.25);
  setPageDownload(tabId, { blob: new Blob(["%PDF"]), name: "paper.pdf", contentType: "application/pdf" });
  fitZoomToPage(tabId, true);
  assert.equal(zoom(), 1);
  const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
  assert.equal(tab && homeZoom(tab), 1);
  setPageDownload(tabId, null);
  fitZoomToPage(tabId, false);
  assert.equal(zoom(), 1.25);
  useBrowserPrefsStore.getState().setDefaultZoom(1);
});

test("expired visits take their sites' icons with them", () => {
  const now = Date.now();
  useBrowserPrefsStore.getState().setHistoryRetentionDays(0);
  useBrowserHistoryStore.setState({
    history: [
      { id: "new", url: "https://www.kept.com/a", title: "Kept", visitedAt: now - DAY_MS },
      { id: "old", url: "https://gone.com/b", title: "Gone", visitedAt: now - 40 * DAY_MS },
    ],
    icons: { "kept.com": "https://kept.com/icon.png", "gone.com": "https://gone.com/icon.png" },
  });
  useBrowserPrefsStore.getState().setHistoryRetentionDays(30);
  assert.deepEqual(useBrowserHistoryStore.getState().icons, { "kept.com": "https://kept.com/icon.png" });
  useBrowserPrefsStore.getState().setHistoryRetentionDays(0);
});

test("a page opened from a new tab uses the current default zoom", async () => {
  const { useBrowserStore } = await import("../src/features/browser/store.ts");
  useBrowserStore.getState().newTab();
  const tabId = useBrowserStore.getState().activeTabId ?? "";
  useBrowserPrefsStore.getState().setDefaultZoom(1.5);
  useBrowserStore.getState().navigate(tabId, { url: "https://example.com/" });
  assert.equal(useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.zoom, 1.5);
  useBrowserStore.getState().setZoom(tabId, 2);
  useBrowserStore.getState().navigate(tabId, { url: "https://example.org/" });
  assert.equal(useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.zoom, 2, "a zoomed page keeps it");
  useBrowserPrefsStore.getState().setDefaultZoom(1);
});

test("an import past the bookmark limit says how many it left out", async () => {
  const { MAX_BOOKMARKS } = await import("../src/features/browser/bookmarks-store.ts");
  useBrowserBookmarksStore.setState({ bookmarks: [] });
  const items = Array.from({ length: MAX_BOOKMARKS + 3 }, (_, index) => ({
    url: `https://example.com/${index}`,
    title: `Page ${index}`,
    folder: "other" as const,
  }));
  // A repeat of one left out isn't counted twice.
  items.push({ url: `https://example.com/${MAX_BOOKMARKS + 1}`, title: "Again", folder: "other" });
  assert.deepEqual(useBrowserBookmarksStore.getState().importBookmarks(items), {
    added: MAX_BOOKMARKS,
    leftOut: 3,
  });
  useBrowserBookmarksStore.setState({ bookmarks: [] });
});

test("back to a file shows it at 100%, forward to the web page at the default zoom", async () => {
  const { useBrowserStore } = await import("../src/features/browser/store.ts");
  const store = useBrowserStore.getState();
  store.openFile({ blob: new Blob(["# notes"]), name: "notes.md", contentType: "text/markdown" });
  const tabId = useBrowserStore.getState().activeTabId ?? "";
  const zoom = () => useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.zoom;
  useBrowserPrefsStore.getState().setDefaultZoom(1.5);
  store.navigate(tabId, { url: "https://example.com/" });
  assert.equal(zoom(), 1.5);
  store.goBack(tabId);
  assert.equal(zoom(), 1);
  store.goForward(tabId);
  assert.equal(zoom(), 1.5);
  store.setZoom(tabId, 2);
  store.goBack(tabId);
  assert.equal(zoom(), 2, "a zoom the reader chose carries over");
  useBrowserPrefsStore.getState().setDefaultZoom(1);
});

test("back to a new tab and forward again keeps the page's zoom", async () => {
  const { useBrowserStore } = await import("../src/features/browser/store.ts");
  const store = useBrowserStore.getState();
  store.newTab();
  const tabId = useBrowserStore.getState().activeTabId ?? "";
  const zoom = () => useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.zoom;
  useBrowserPrefsStore.getState().setDefaultZoom(1.25);
  store.navigate(tabId, { url: "https://example.com/" });
  assert.equal(zoom(), 1.25);
  store.setZoom(tabId, 2);
  store.goBack(tabId);
  store.goForward(tabId);
  assert.equal(zoom(), 2);
  useBrowserPrefsStore.getState().setDefaultZoom(1);
});

test("a new site's icon survives the prune its first visit triggers", () => {
  const now = Date.now();
  useBrowserPrefsStore.getState().setHistoryRetentionDays(30);
  useBrowserHistoryStore.setState({
    history: [{ id: "old", url: "https://gone.com/", title: "Gone", visitedAt: now - 40 * DAY_MS }],
    icons: { "gone.com": "https://gone.com/icon.png" },
  });
  // As a page load does: the icon first, then the visit.
  useBrowserHistoryStore.getState().recordIcon("fresh.com", "https://fresh.com/icon.png");
  useBrowserHistoryStore.getState().recordVisit("https://fresh.com/", "Fresh");
  assert.deepEqual(useBrowserHistoryStore.getState().icons, { "fresh.com": "https://fresh.com/icon.png" });
  useBrowserPrefsStore.getState().setHistoryRetentionDays(0);
});

test("a default zoom chosen while a fetched file shows applies to the next page", async () => {
  const { useBrowserStore, setPageDownload } = await import("../src/features/browser/store.ts");
  const { fitZoomToPage } = await import("../src/features/browser/zoom.ts");
  useBrowserPrefsStore.getState().setDefaultZoom(1);
  useBrowserStore.getState().openUrl("https://example.com/report.pdf", { newTab: true });
  const tabId = useBrowserStore.getState().activeTabId ?? "";
  const zoom = () => useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.zoom;
  setPageDownload(tabId, { blob: new Blob(["%PDF"]), name: "report.pdf", contentType: "application/pdf" });
  fitZoomToPage(tabId, true);
  assert.equal(zoom(), 1);
  useBrowserPrefsStore.getState().setDefaultZoom(1.5);
  setPageDownload(tabId, null);
  fitZoomToPage(tabId, false);
  assert.equal(zoom(), 1.5);
  useBrowserPrefsStore.getState().setDefaultZoom(1);
});

test("a page the reader set to 100% keeps it after a fetched file", async () => {
  const { useBrowserStore, setPageDownload } = await import("../src/features/browser/store.ts");
  const { fitZoomToPage } = await import("../src/features/browser/zoom.ts");
  useBrowserPrefsStore.getState().setDefaultZoom(1.5);
  useBrowserStore.getState().openUrl("https://example.com/a", { newTab: true });
  const tabId = useBrowserStore.getState().activeTabId ?? "";
  const zoom = () => useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.zoom;
  useBrowserStore.getState().setZoom(tabId, 1);
  setPageDownload(tabId, { blob: new Blob(["%PDF"]), name: "b.pdf", contentType: "application/pdf" });
  fitZoomToPage(tabId, true);
  setPageDownload(tabId, null);
  fitZoomToPage(tabId, false);
  assert.equal(zoom(), 1);
  useBrowserPrefsStore.getState().setDefaultZoom(1);
});
