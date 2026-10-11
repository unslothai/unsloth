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
const { useChatRuntimeStore } = await import("@/features/chat");
const { currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");

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

test("pages visited beside a temporary chat stay out of history, icons included", () => {
  const history = useBrowserHistoryStore.getState();
  history.clearHistory();
  useChatRuntimeStore.getState().setIncognito(true);
  history.recordIcon("example.com", "https://example.com/icon.png");
  history.recordVisit("https://example.com/", "Example");
  assert.deepEqual(useBrowserHistoryStore.getState().history, []);
  assert.deepEqual(useBrowserHistoryStore.getState().icons, {});
  useChatRuntimeStore.getState().setIncognito(false);
  history.recordVisit("https://example.com/", "Example");
  assert.equal(useBrowserHistoryStore.getState().history.length, 1);
});

test("a page opened beside a temporary chat stays out of history when it loads after the chat turns normal", () => {
  const history = useBrowserHistoryStore.getState();
  history.clearHistory();
  useChatRuntimeStore.getState().setIncognito(true);
  useBrowserStore.getState().openUrl("https://example.com/", { newTab: true });
  const tab = useBrowserStore.getState().tabs.find((item) => item.id === useBrowserStore.getState().activeTabId);
  const entry = tab ? currentEntry(tab) : null;
  useChatRuntimeStore.getState().setIncognito(false);
  assert.equal(entry?.kind === "web" && entry.temporary, true);
  history.recordVisit("https://example.com/", "Example", entry?.kind === "web" ? entry.temporary : false);
  assert.deepEqual(useBrowserHistoryStore.getState().history, []);
  useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
  const next = useBrowserStore.getState().tabs.find((item) => item.id === useBrowserStore.getState().activeTabId);
  const nextEntry = next ? currentEntry(next) : null;
  history.recordVisit("https://example.org/", "Example", nextEntry?.kind === "web" ? nextEntry.temporary : false);
  assert.equal(useBrowserHistoryStore.getState().history.length, 1);
});

test("with download history off, downloads are not listed", () => {
  const history = useBrowserHistoryStore.getState();
  history.clearDownloads();
  useBrowserPrefsStore.getState().setSaveDownloadHistory(false);
  assert.equal(history.recordDownload(download), undefined);
  assert.equal(useBrowserHistoryStore.getState().downloads.length, 0);
  useBrowserPrefsStore.getState().setSaveDownloadHistory(true);
  const id = history.recordDownload(download);
  assert.equal(useBrowserHistoryStore.getState().downloads.length, 1);
  assert.equal(useBrowserHistoryStore.getState().downloads[0].id, id);
});

test("a navigation asked for as temporary stays temporary in a normal chat", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://example.com/", { newTab: true });
  const tabId = useBrowserStore.getState().activeTabId!;
  store.navigate(tabId, { url: "https://example.com/reached", temporary: true });
  const entry = currentEntry(useBrowserStore.getState().tabs.find((item) => item.id === tabId)!);
  assert.equal(entry.kind === "web" && entry.temporary, true);
  store.navigate(tabId, { url: "https://example.com/after" });
  const after = currentEntry(useBrowserStore.getState().tabs.find((item) => item.id === tabId)!);
  assert.equal(after.kind === "web" && after.temporary, undefined);
  // A page noted as normal stays normal when it is kept during a temporary chat.
  useChatRuntimeStore.getState().setIncognito(true);
  store.navigate(tabId, { url: "https://example.com/kept", temporary: false });
  useChatRuntimeStore.getState().setIncognito(false);
  const kept = currentEntry(useBrowserStore.getState().tabs.find((item) => item.id === tabId)!);
  assert.equal(kept.kind === "web" && kept.temporary, undefined);
  store.closeTab(tabId);
});

test("files downloaded beside a temporary chat are not listed", () => {
  const history = useBrowserHistoryStore.getState();
  history.clearDownloads();
  useChatRuntimeStore.getState().setIncognito(true);
  history.recordDownload(download);
  assert.deepEqual(useBrowserHistoryStore.getState().downloads, []);
  useChatRuntimeStore.getState().setIncognito(false);
  history.recordDownload(download);
  assert.equal(useBrowserHistoryStore.getState().downloads.length, 1);
});

test("a download begun beside a temporary chat stays unlisted when it is saved after the chat turns normal", async () => {
  const { saveBrowserDownload } = await import("../src/features/browser/downloads.ts");
  const history = useBrowserHistoryStore.getState();
  history.clearDownloads();
  const prefs = useBrowserPrefsStore.getState();
  prefs.setAskWhereToSave(true);
  prefs.setAskBeforeDownloading(false);
  const g = globalThis as { showSaveFilePicker?: unknown; __toasts?: { options?: { action?: { onClick: () => void } } }[] };
  const written: Blob[] = [];
  g.showSaveFilePicker = async ({ suggestedName }: { suggestedName: string }) => ({
    name: suggestedName,
    createWritable: async () => ({ write: async (data: Blob) => void written.push(data), close: async () => {} }),
  });
  Object.defineProperty(globalThis, "navigator", { value: { userActivation: { isActive: false } }, configurable: true });
  g.__toasts = [];
  try {
    useChatRuntimeStore.getState().setIncognito(true);
    await saveBrowserDownload({ blob: new Blob(["x"]), name: "late.zip", contentType: "application/zip", url: "https://a.example/late.zip" });
    useChatRuntimeStore.getState().setIncognito(false);
    g.__toasts.at(-1)!.options!.action!.onClick();
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(written.length, 1);
    assert.deepEqual(useBrowserHistoryStore.getState().downloads, []);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    prefs.setAskWhereToSave(false);
    prefs.setAskBeforeDownloading(true);
    delete g.showSaveFilePicker;
  }
});

test("a file a temporary chat's page fetched stays unlisted when it is saved after the chat turns normal", async () => {
  const { saveBrowserDownload } = await import("../src/features/browser/downloads.ts");
  useBrowserHistoryStore.getState().clearDownloads();
  const written: Blob[] = [];
  const target = {
    name: "late.bin",
    createWritable: async () => ({ write: async (data: Blob) => void written.push(data), close: async () => {} }),
  };
  const download = { blob: new Blob(["x"]), name: "late.bin", contentType: "application/octet-stream", url: "https://a.example/late.bin" };
  await saveBrowserDownload({ ...download, temporary: true }, target);
  assert.equal(written.length, 1);
  assert.deepEqual(useBrowserHistoryStore.getState().downloads, []);
  await saveBrowserDownload(download, target);
  assert.equal(useBrowserHistoryStore.getState().downloads.length, 1);
  // Taken as normal before a temporary chat was shown: still listed.
  useChatRuntimeStore.getState().setIncognito(true);
  try {
    await saveBrowserDownload({ ...download, temporary: false }, target);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
  }
  assert.equal(useBrowserHistoryStore.getState().downloads.length, 2);
});

test("Save link as from a temporary chat stays unlisted when its fetch lands after the chat turns normal", async () => {
  const { saveLinkAs } = await import("../src/features/browser/downloads.ts");
  useBrowserHistoryStore.getState().clearDownloads();
  const prefs = useBrowserPrefsStore.getState();
  prefs.setAskWhereToSave(true);
  prefs.setAskBeforeDownloading(false);
  const g = globalThis as { showSaveFilePicker?: unknown; __authFetch?: unknown };
  const written: Blob[] = [];
  g.showSaveFilePicker = async ({ suggestedName }: { suggestedName: string }) => ({
    name: suggestedName,
    createWritable: async () => ({ write: async (data: Blob) => void written.push(data), close: async () => {} }),
  });
  g.__authFetch = () =>
    new Promise((resolve) =>
      setTimeout(
        () =>
          resolve(
            new Response(new Blob(["x"]), {
              headers: { "Content-Type": "application/zip", "X-Unsloth-Browser-Filename": "data.zip" },
            }),
          ),
        50,
      ),
    );
  try {
    useChatRuntimeStore.getState().setIncognito(true);
    const saving = saveLinkAs("https://a.example/data.zip");
    useChatRuntimeStore.getState().setIncognito(false);
    await saving;
    assert.equal(written.length, 1);
    assert.deepEqual(useBrowserHistoryStore.getState().downloads, []);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    prefs.setAskWhereToSave(false);
    prefs.setAskBeforeDownloading(true);
    delete g.showSaveFilePicker;
    delete g.__authFetch;
  }
});

test("a download begun beside a normal chat is listed though it lands while a temporary chat is shown", () => {
  const history = useBrowserHistoryStore.getState();
  history.clearDownloads();
  useChatRuntimeStore.getState().setIncognito(true);
  history.recordDownload(download, false);
  useChatRuntimeStore.getState().setIncognito(false);
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

test("a site's download answer is remembered, changed and forgotten, per account", async () => {
  const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
  const { accountDatabaseName } = await import("../src/lib/account-transition.ts");
  const sites = useDownloadSitesStore.getState();
  sites.setSite("example.com", "allow");
  sites.setSite("example.org", "block");
  assert.deepEqual(useDownloadSitesStore.getState().sites, { "example.com": "allow", "example.org": "block" });
  sites.setSite("example.com", "block");
  sites.setSite("example.org", null);
  assert.deepEqual(useDownloadSitesStore.getState().sites, { "example.com": "block" });
  const name = useDownloadSitesStore.persist.getOptions().name;
  assert.equal(name, accountDatabaseName("unsloth_browser_download_sites"));
  assert.notEqual(name, "unsloth_browser_download_sites");
  assert.ok(data.has(name!));
  const prefs = JSON.parse(data.get("unsloth_browser_prefs") ?? "{}") as { state?: Record<string, unknown> };
  assert.equal(prefs.state?.downloadSites, undefined);
  assert.equal(useBrowserPrefsStore.getState().askBeforeDownloading, true);
  sites.setSite("example.com", null);
});

test("a remembered answer settles the site's other waiting downloads", async () => {
  const { approveDownload, answerDownload, useApprovalStore } = await import(
    "../src/features/browser/download-approval-queue.ts"
  );
  const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
  const first = approveDownload("https://a.example/1.zip", "1.zip");
  const second = approveDownload("https://a.example/2.zip", "2.zip");
  const other = approveDownload("https://b.example/3.zip", "3.zip");
  answerDownload(useApprovalStore.getState().queue[0], false, true);
  assert.equal(await first, false);
  assert.equal(await second, false);
  assert.deepEqual(useApprovalStore.getState().queue.map((request) => request.origin), ["https://b.example"]);
  assert.equal(useDownloadSitesStore.getState().sites["https://a.example"], "block");
  const fourth = approveDownload("https://b.example/4.zip", "4.zip");
  answerDownload(useApprovalStore.getState().queue[0], true, false);
  assert.equal(await other, true);
  assert.equal(useApprovalStore.getState().queue.length, 1);
  answerDownload(useApprovalStore.getState().queue[0], true, false);
  assert.equal(await fourth, true);
  assert.equal(useDownloadSitesStore.getState().sites["https://b.example"], undefined);
  useDownloadSitesStore.getState().setSite("https://a.example", null);
});

test("a context menu download skips a site's block, and a file that runs code still asks", async () => {
  const { approveChosenDownload, answerDownload, useApprovalStore } = await import(
    "../src/features/browser/download-approval-queue.ts"
  );
  const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
  useDownloadSitesStore.getState().setSite("https://e.example", "block");
  try {
    assert.equal(await approveChosenDownload("https://e.example/cat.png", "cat.png"), true);
    const setup = approveChosenDownload("https://e.example/setup.exe", "setup.exe");
    // No origin: the prompt offers nothing to remember.
    assert.deepEqual(
      useApprovalStore.getState().queue.map(({ origin, label, dangerous }) => [origin, label, dangerous]),
      [["", "e.example", true]],
    );
    answerDownload(useApprovalStore.getState().queue[0], true, true);
    assert.equal(await setup, true);
    assert.equal(useDownloadSitesStore.getState().sites["https://e.example"], "block");
  } finally {
    useDownloadSitesStore.getState().setSite("https://e.example", null);
  }
});

test("a file that runs code asks whatever the site's remembered answer or the ask setting", async () => {
  const { approveDownload, answerDownload, useApprovalStore } = await import(
    "../src/features/browser/download-approval-queue.ts"
  );
  const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
  useDownloadSitesStore.getState().setSite("https://c.example", "allow");
  assert.equal(await approveDownload("https://c.example/notes.pdf", "notes.pdf"), true);
  const setup = approveDownload("https://c.example/setup.exe", "setup.exe");
  assert.deepEqual(
    useApprovalStore.getState().queue.map(({ name, dangerous }) => [name, dangerous]),
    [["setup.exe", true]],
  );
  answerDownload(useApprovalStore.getState().queue[0], false, false);
  assert.equal(await setup, false);
  useDownloadSitesStore.getState().setSite("https://c.example", null);

  useBrowserPrefsStore.getState().setAskBeforeDownloading(false);
  try {
    assert.equal(await approveDownload("https://d.example/a.zip", "a.zip"), true);
    // Bidi controls are shown as "_", so the name can't read as another type.
    const disguised = approveDownload("https://d.example/x", "invoice\u202efdp.exe");
    assert.equal(useApprovalStore.getState().queue[0]?.name, "invoice_fdp.exe");
    answerDownload(useApprovalStore.getState().queue[0], true, false);
    assert.equal(await disguised, true);
  } finally {
    useBrowserPrefsStore.getState().setAskBeforeDownloading(true);
  }

  // Remembering Allow on one file settles the site's other waiting files, except ones that run code.
  const first = approveDownload("https://e.example/1.zip", "1.zip");
  const tool = approveDownload("https://e.example/tool.msi", "tool.msi");
  const second = approveDownload("https://e.example/2.zip", "2.zip");
  answerDownload(useApprovalStore.getState().queue[0], true, true);
  assert.equal(await first, true);
  assert.equal(await second, true);
  assert.deepEqual(
    useApprovalStore.getState().queue.map(({ name }) => name),
    ["tool.msi"],
  );
  // A later one still asks.
  const later = approveDownload("https://e.example/run.bat", "run.bat");
  assert.equal(useApprovalStore.getState().queue.length, 2);
  answerDownload(useApprovalStore.getState().queue[0], true, false);
  answerDownload(useApprovalStore.getState().queue[0], false, false);
  assert.equal(await tool, true);
  assert.equal(await later, false);
  useDownloadSitesStore.getState().setSite("https://e.example", null);
});

test("files that run code are recognised however the name is cased, padded or pathed", async () => {
  const { readFileSync } = await import("node:fs");
  const { isDangerousDownload } = await import("../src/features/browser/download-safety.ts");
  const fixture = JSON.parse(
    readFileSync(new URL("./fixtures/dangerous-download-names.json", import.meta.url), "utf8"),
  ) as { cases: { name: string; dangerous: boolean }[] };
  assert.ok(fixture.cases.length > 50);
  for (const { name, dangerous } of fixture.cases) assert.equal(isDangerousDownload(name), dangerous, name);
  assert.equal(isDangerousDownload("C:\\Users\\a\\setup.exe"), true);
  assert.equal(isDangerousDownload("dir.exe/readme.txt"), false);
  // Saved names are cut to 240 bytes: a tail too long to keep as the extension is cut off, leaving .exe last.
  assert.equal(isDangerousDownload(`${"a".repeat(236)}.exe${"x".repeat(40)}`), true);
  assert.equal(isDangerousDownload(`${"a".repeat(300)}.txt`), false);
  assert.equal(isDangerousDownload(`${"a".repeat(236)}.txt${"x".repeat(40)}`), false);
  // A Windows device name gets a leading "_", which moves the cut by one byte.
  assert.equal(isDangerousDownload(`con.${"a".repeat(231)}.exe${"x".repeat(40)}`), true);
  assert.equal(isDangerousDownload(`cox.${"a".repeat(231)}.exe${"x".repeat(40)}`), false);
  // A decoded "dir/" before the name: the desktop app may save only the last part, cut to end in .exe.
  assert.equal(isDangerousDownload(`dir/${"a".repeat(236)}.exe${"x".repeat(40)}`), true);
  // Controls are one-byte "_" when saved, so two-byte C1 controls move the cut.
  assert.equal(isDangerousDownload(`${"\u0085".repeat(236)}.exe${"x".repeat(40)}`), true);
});

test("download answers are kept per origin, never for every site", async () => {
  const { downloadSiteOf: siteOf } = await import("../src/features/browser/download-approval-queue.ts");
  // A blob: URL is its creator's; an opaque one, or data: and about:, belongs to no site.
  assert.equal(siteOf("blob:https://a.example/uuid"), "https://a.example");
  assert.equal(siteOf("blob:null/uuid"), "");
  assert.equal(siteOf("data:text/plain,x"), "");
  assert.equal(siteOf("about:blank"), "");
  const { approveDownload, answerDownload, useApprovalStore } = await import(
    "../src/features/browser/download-approval-queue.ts"
  );
  const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
  const blob = approveDownload("blob:null/uuid", "file.bin", "blob:null/uuid");
  const request = useApprovalStore.getState().queue[0];
  assert.equal(request.origin, "");
  answerDownload(request, true, true);
  assert.equal(await blob, true);
  assert.deepEqual(useDownloadSitesStore.getState().sites, {});
  const paged = approveDownload("blob:https://cdn.example/uuid", "file.bin", "https://a.example/page");
  assert.equal(useApprovalStore.getState().queue[0].origin, "https://a.example");
  answerDownload(useApprovalStore.getState().queue[0], true, true);
  assert.equal(await paged, true);
  assert.deepEqual(useDownloadSitesStore.getState().sites, { "https://a.example": "allow" });
  // Another scheme, port or subdomain is another site: it is asked about again.
  const { downloadSiteOf } = await import("../src/features/browser/download-approval-queue.ts");
  assert.equal(downloadSiteOf("https://www.a.example/x"), "https://www.a.example");
  assert.equal(downloadSiteOf("http://a.example:8443/x"), "http://a.example:8443");
  const other = approveDownload("http://a.example:8443/f.zip", "f.zip");
  assert.equal(useApprovalStore.getState().queue.length, 1);
  answerDownload(useApprovalStore.getState().queue[0], false, false);
  assert.equal(await other, false);
  assert.equal(await approveDownload("https://a.example/g.zip", "g.zip"), true);
  useDownloadSitesStore.getState().setSite("https://a.example", null);
});

test("a download whose click has expired waits for Save rather than skip the save dialog", async () => {
  const picked: string[] = [];
  const written: Blob[] = [];
  let active = false;
  // Chromium's picker refuses to open without a recent click.
  Object.assign(globalThis, {
    showSaveFilePicker: async ({ suggestedName }: { suggestedName: string }) => {
      if (!active) throw new DOMException("Must be handling a user gesture", "SecurityError");
      picked.push(suggestedName);
      return {
        name: suggestedName,
        createWritable: async () => ({ write: async (data: Blob) => void written.push(data), close: async () => {} }),
      };
    },
  });
  Object.defineProperty(globalThis, "navigator", { value: { userActivation: { get isActive() { return active; } } }, configurable: true });
  const { saveBrowserDownload } = await import("../src/features/browser/downloads.ts");
  const prefs = useBrowserPrefsStore.getState();
  prefs.setAskWhereToSave(true);
  prefs.setAskBeforeDownloading(false);
  (globalThis as { __toasts?: unknown[] }).__toasts = [];
  const blob = new Blob(["x"]);
  // Not a file that runs code: those ask first whatever the setting.
  await saveBrowserDownload({ blob, name: "data.zip", contentType: "application/zip", url: "https://a.example/data.zip" });
  const toasts = (globalThis as { __toasts?: { message: string; options?: { action?: { onClick: () => void } } }[] }).__toasts!;
  assert.equal(toasts.length, 1);
  assert.equal(toasts[0].message, "browser.downloadPrompt.ready");
  assert.deepEqual(picked, []);
  active = true;
  toasts[0].options!.action!.onClick();
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.deepEqual(picked, ["data.zip"]);
  assert.equal(written.length, 1);
  prefs.setAskWhereToSave(false);
  prefs.setAskBeforeDownloading(true);
  delete (globalThis as { showSaveFilePicker?: unknown }).showSaveFilePicker;
});

test("a file a page sends the tab to is asked about for that page, not the file's site", async () => {
  const { useBrowserStore, currentEntry } = await import("../src/features/browser/store.ts");
  const { saveBrowserDownload } = await import("../src/features/browser/downloads.ts");
  const { answerDownload, useApprovalStore } = await import("../src/features/browser/download-approval-queue.ts");
  const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
  const store = useBrowserStore.getState();
  store.openUrl("https://a.example/page", { newTab: true });
  const tabId = useBrowserStore.getState().activeTabId!;
  store.navigate(tabId, { url: "https://b.example/setup.zip", from: "https://a.example/page" });
  const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId)!;
  const entry = currentEntry(tab);
  assert.ok(entry.kind === "web");
  assert.equal(entry.from, "https://a.example/page");
  store.navigate(tabId, { url: "https://b.example/other.zip" });
  const typed = currentEntry(useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId)!);
  assert.ok(typed.kind === "web" && typed.from === undefined);

  // b.example is trusted, but a.example sent the tab there: still asked, for a.example.
  useDownloadSitesStore.getState().setSite("https://b.example", "allow");
  const blob = new Blob(["x"]);
  const saving = saveBrowserDownload({ blob, name: "setup.zip", contentType: "application/zip", url: "https://b.example/setup.zip", site: entry.from });
  await new Promise((resolve) => setTimeout(resolve, 0));
  const queue = useApprovalStore.getState().queue;
  assert.equal(queue.length, 1);
  assert.equal(queue[0].origin, "https://a.example");
  answerDownload(queue[0], false, false);
  await saving;
  useDownloadSitesStore.getState().setSite("https://b.example", null);
  store.closeTab(tabId);
});

test("Save link as asks again when the file it named by the address turns out to run code", async () => {
  const { saveLinkAs } = await import("../src/features/browser/downloads.ts");
  const { answerDownload, useApprovalStore } = await import("../src/features/browser/download-approval-queue.ts");
  const prefs = useBrowserPrefsStore.getState();
  prefs.setAskWhereToSave(true);
  prefs.setAskBeforeDownloading(false);
  const g = globalThis as { showSaveFilePicker?: unknown; __authFetch?: unknown };
  // The picker fails without a cancel, so the save falls back to a plain download by the server's name.
  g.showSaveFilePicker = async () => {
    throw new DOMException("no picker here", "NotAllowedError");
  };
  // Slower than the wait before asking, so the address's name ("download") is the one approved first.
  g.__authFetch = () =>
    new Promise((resolve) =>
      setTimeout(
        () =>
          resolve(
            new Response(new Blob(["MZ"]), {
              headers: { "Content-Type": "application/octet-stream", "X-Unsloth-Browser-Filename": "setup.exe" },
            }),
          ),
        1200,
      ),
    );
  try {
    const saving = saveLinkAs("https://a.example/download");
    for (let i = 0; i < 40 && useApprovalStore.getState().queue.length === 0; i++) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    const asked = useApprovalStore.getState().queue[0];
    assert.equal(asked?.name, "setup.exe");
    assert.equal(asked?.dangerous, true);
    answerDownload(asked, false, false);
    await saving;
    assert.equal(useApprovalStore.getState().queue.length, 0);
  } finally {
    delete g.showSaveFilePicker;
    delete g.__authFetch;
    prefs.setAskWhereToSave(false);
    prefs.setAskBeforeDownloading(true);
  }
});

test("a blocked site's blob: page can't download past the block", async () => {
  const { approveDownload } = await import("../src/features/browser/download-approval-queue.ts");
  const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
  const prefs = useBrowserPrefsStore.getState();
  prefs.setAskBeforeDownloading(false);
  useDownloadSitesStore.getState().setSite("https://a.example", "block");
  assert.equal(await approveDownload("blob:https://a.example/f", "f.zip", "blob:https://a.example/doc"), false);
  assert.equal(await approveDownload("https://cdn.example/f.zip", "f.zip", "https://b.example/"), true);
  useDownloadSitesStore.getState().setSite("https://a.example", null);
  prefs.setAskBeforeDownloading(true);
});

test("the download prompt offers Remember only for files a remembered answer can cover", async () => {
  const { readFileSync } = await import("node:fs");
  const source = readFileSync(new URL("../src/features/browser/download-approval.tsx", import.meta.url), "utf8");
  // A file that runs code asks whatever was remembered (approveDownload), so its prompt has no checkbox.
  assert.match(source, /\{request\?\.origin && !request\.dangerous \? \(\s*<label/);
});

test("a finished download links to its history row, and an unrecorded one drops its forgotten native id", async () => {
  const { finishDownload, keptDownloadFile, mountDownloadsButton, useDownloadActivity } = await import(
    "../src/features/browser/download-activity.ts"
  );
  const file = { blob: new Blob(["x"]), name: "x".repeat(300), contentType: "text/plain" };
  const result = { name: file.name, size: 1, contentType: "text/plain", url: null, failed: false };
  const unmount = mountDownloadsButton();
  try {
    // A name past the history row's limit still links: the row's id comes from recordDownload.
    finishDownload("save:1", { ...result, nativeId: "n1", historyId: "row1" }, file);
    assert.equal(useDownloadActivity.getState().finished?.historyId, "row1");
    assert.equal(useDownloadActivity.getState().finished?.nativeId, "n1");
    assert.equal(keptDownloadFile("row1"), file);
    // History off: the app forgot the native id, so Open and Show in folder can't use it.
    finishDownload("save:2", { ...result, nativeId: "n2" });
    assert.equal(useDownloadActivity.getState().finished?.nativeId, undefined);
  } finally {
    useDownloadActivity.getState().dismissFinished();
    unmount();
  }
  // No button on screen: the caller toasts it, so nothing holds the result.
  finishDownload("save:3", result);
  assert.equal(useDownloadActivity.getState().finished, null);
});

test("a long download name keeps its extension, and a kept copy goes with its history row", async () => {
  const { finishDownload, keptDownloadFile } = await import("../src/features/browser/download-activity.ts");
  const { isDangerousDownload } = await import("../src/features/browser/download-safety.ts");
  const history = useBrowserHistoryStore.getState();
  history.clearDownloads();
  const id = history.recordDownload({ ...download, name: `${"a".repeat(210)}.exe` });
  const row = useBrowserHistoryStore.getState().downloads[0];
  assert.equal(row.name.length, 200);
  assert.ok(row.name.endsWith(".exe"));
  assert.ok(isDangerousDownload(row.name));
  const file = { blob: new Blob(["x"]), name: row.name, contentType: "" };
  finishDownload("save:row", { name: row.name, size: 1, contentType: "", url: null, historyId: id, failed: false }, file);
  assert.equal(keptDownloadFile(id), file);
  useBrowserHistoryStore.getState().removeDownload(id as string);
  assert.equal(keptDownloadFile(id), null);
});

test("an unrecorded result's bytes go with its notice, and one replaced on screen is toasted", async () => {
  const { finishDownload, keptDownloadFile, mountDownloadsButton, useDownloadActivity } = await import(
    "../src/features/browser/download-activity.ts"
  );
  const file = { blob: new Blob(["x"]), name: "a.bin", contentType: "" };
  const result = { name: "a.bin", size: 1, contentType: "", url: null, failed: false };
  // No button on screen: an unrecorded copy has no way back, so it isn't kept; a recorded one is.
  finishDownload("save:z", result, file);
  assert.equal(keptDownloadFile("save:z"), null);
  finishDownload("save:y", { ...result, historyId: "row-y" }, file);
  assert.equal(keptDownloadFile("row-y"), file);
  const unmount = mountDownloadsButton();
  const toasts = ((globalThis as { __toasts?: { message: string; options?: { description?: string } }[] }).__toasts ??=
    []);
  toasts.length = 0;
  try {
    finishDownload("save:a", result, file);
    assert.equal(keptDownloadFile("save:a"), file);
    useDownloadActivity.getState().dismissFinished();
    assert.equal(keptDownloadFile("save:a"), null);
    // A failure the list never shows, replaced before it was dismissed: toasted, not lost.
    finishDownload("native:p1", { ...result, name: "lost.zip", failed: true });
    finishDownload("save:b", result, file);
    assert.deepEqual(
      toasts.map((item) => [item.message, item.options?.description]),
      [["browser.downloads.failed", "lost.zip"]],
    );
    assert.equal(keptDownloadFile("save:b"), file);
    // A recorded result replaced on screen stays in the list, so no toast.
    finishDownload("save:c", { ...result, historyId: "row-c" });
    finishDownload("save:d", { ...result, historyId: "row-d" });
    assert.equal(toasts.length, 2);
    assert.equal(keptDownloadFile("save:b"), null);
  } finally {
    useDownloadActivity.getState().dismissFinished();
    unmount();
  }
});

test("video and audio tabs don't zoom; pages, images and documents do", async () => {
  const { canZoom } = await import("../src/features/browser/zoom.ts");
  const tab = (entry: object) =>
    ({ id: "t", index: 0, history: [{ kind: "file", contentType: "", ...entry }], zoom: 1 }) as never;
  assert.equal(canZoom(tab({ name: "clip.mp4" })), false);
  assert.equal(canZoom(tab({ name: "clip", contentType: "video/webm" })), false);
  assert.equal(canZoom(tab({ name: "song.mp3" })), false);
  assert.equal(canZoom(tab({ name: "photo.png" })), true);
  assert.equal(canZoom(tab({ name: "paper.pdf" })), true);
  // A clip shown as its text is a text file.
  assert.equal(canZoom(tab({ name: "clip.mp4", plainText: true })), true);
  assert.equal(canZoom({ id: "t", index: 0, history: [{ kind: "web", url: "https://a.b/" }], zoom: 1 } as never), true);
});

test("closing the last Downloads button dismisses its result, but a button swapped in keeps it", async () => {
  const { finishDownload, keptDownloadFile, mountDownloadsButton, useDownloadActivity } = await import(
    "../src/features/browser/download-activity.ts"
  );
  const file = { blob: new Blob(["x"]), name: "a.bin", contentType: "" };
  const result = { name: "a.bin", size: 1, contentType: "", url: null, failed: false };
  // A tab change: one button goes and another comes in the same commit.
  const first = mountDownloadsButton();
  finishDownload("save:swap", result, file);
  first();
  const second = mountDownloadsButton();
  await Promise.resolve();
  assert.equal(useDownloadActivity.getState().finished?.key, "save:swap");
  assert.equal(keptDownloadFile("save:swap"), file);
  // The panel closes: no button is back, so the unrecorded copy goes.
  second();
  await Promise.resolve();
  assert.equal(useDownloadActivity.getState().finished, null);
  assert.equal(keptDownloadFile("save:swap"), null);
});

test("file toolbars keep a Downloads button registered, and a hidden panel closes its popover", async () => {
  const { readFileSync } = await import("node:fs");
  const read = (file: string) => readFileSync(new URL(`../src/features/browser/${file}`, import.meta.url), "utf8");
  // Mounted only once a save began, it would register after a quick save had already finished.
  const panel = read("browser-panel.tsx");
  assert.equal(panel.match(/<DownloadsButton [^>]*visible=\{visible\} idleHidden=\{true\} \/>/g)?.length, 3);
  assert.doesNotMatch(panel, /BusyDownloadsButton/);
  // The popover is portaled out of the panel, so it would stay up over the next page.
  assert.match(read("downloads-button.tsx"), /if \(!visible && mode !== "closed"\) \{\s*setMode\("closed"\);/);
});
