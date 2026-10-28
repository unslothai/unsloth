// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const data = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
const listeners = new Map<string, ((event: { key: string | null }) => void)[]>();
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: {
    localStorage,
    location: { protocol: "http:" },
    addEventListener: (type: string, listener: (event: { key: string | null }) => void) =>
      listeners.set(type, [...(listeners.get(type) ?? []), listener]),
  },
});

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserHistoryStore } = await import("../src/features/browser/history-store.ts");
const { useBrowserPrefsStore } = await import("../src/features/browser/prefs-store.ts");
const { useBrowserBookmarksStore } = await import("../src/features/browser/bookmarks-store.ts");
const { usePinnedPagesStore } = await import("../src/features/browser/pinned-pages-store.ts");
const { useDownloadSitesStore } = await import("../src/features/browser/download-sites-store.ts");
const { useBrowserAgentStore, patchAgentChat } = await import("../src/features/browser/agent/agent-store.ts");

function stored(key: string): Record<string, unknown> {
  return JSON.parse(data.get(key) ?? "{}").state ?? {};
}

function otherWindowWrites(key: string, change: (state: Record<string, unknown>) => Record<string, unknown>) {
  const saved = JSON.parse(data.get(key) ?? "{}");
  data.set(key, JSON.stringify({ ...saved, state: change(saved.state ?? {}) }));
  for (const listener of listeners.get("storage") ?? []) listener({ key });
}

const flush = () => {
  for (const listener of listeners.get("pagehide") ?? []) listener({ key: null });
};

test("a setting turned off in another window stays off after this window changes another one", () => {
  useBrowserPrefsStore.getState().setShowFullUrl(false);
  otherWindowWrites("unsloth_browser_prefs", (state) => ({ ...state, saveHistory: false }));
  useBrowserPrefsStore.getState().setShowFullUrl(true);
  assert.equal(stored("unsloth_browser_prefs").saveHistory, false);
  assert.equal(stored("unsloth_browser_prefs").showFullUrl, true);
  useBrowserHistoryStore.getState().recordVisit("https://example.org/", "Org");
  assert.equal(useBrowserHistoryStore.getState().history.length, 0);
  useBrowserPrefsStore.getState().setSaveHistory(true);
});

test("history cleared in another window does not come back from this window's visits", () => {
  useBrowserHistoryStore.getState().recordVisit("https://example.com/", "Example");
  flush();
  otherWindowWrites("unsloth_browser_history:b1", (state) => ({ ...state, history: [], icons: {} }));
  useBrowserHistoryStore.getState().recordVisit("https://httpbin.org/html", "httpbin");
  flush();
  const urls = (stored("unsloth_browser_history:b1").history as { url: string }[]).map((visit) => visit.url);
  assert.deepEqual(urls, ["https://httpbin.org/html"]);
});

test("a write another window makes wins over this window's unsaved one", () => {
  useBrowserHistoryStore.getState().recordVisit("https://example.net/", "Net");
  otherWindowWrites("unsloth_browser_history:b1", (state) => ({ ...state, history: [] }));
  flush();
  assert.deepEqual(stored("unsloth_browser_history:b1").history, []);
});

test("bookmarks, pinned pages and download site rules saved in two windows are all kept", () => {
  useBrowserBookmarksStore.getState().addBookmark("https://example.com/", "Example");
  otherWindowWrites("unsloth_browser_bookmarks:b1", (state) => ({
    ...state,
    bookmarks: [{ id: "x", url: "https://httpbin.org/", title: "httpbin", folder: "toolbar", addedAt: 1 }],
  }));
  useBrowserBookmarksStore.getState().addBookmark("https://example.org/", "Org");
  const bookmarks = (stored("unsloth_browser_bookmarks:b1").bookmarks as { url: string }[]).map((b) => b.url);
  assert.deepEqual(bookmarks, ["https://httpbin.org/", "https://example.org/"]);

  usePinnedPagesStore.getState().pin("https://example.com/", "Example");
  otherWindowWrites("unsloth_browser_pinned_pages:b1", (state) => ({
    ...state,
    pages: [{ id: "p", url: "https://httpbin.org/", title: "httpbin", pinnedAt: 1 }],
  }));
  usePinnedPagesStore.getState().pin("https://example.org/", "Org");
  const pages = (stored("unsloth_browser_pinned_pages:b1").pages as { url: string }[]).map((p) => p.url);
  assert.deepEqual(pages, ["https://httpbin.org/", "https://example.org/"]);

  useDownloadSitesStore.getState().setSite("https://example.com", "allow");
  otherWindowWrites("unsloth_browser_download_sites:b1", () => ({ sites: { "https://httpbin.org": "block" } }));
  useDownloadSitesStore.getState().setSite("https://example.org", "allow");
  assert.deepEqual(stored("unsloth_browser_download_sites:b1").sites, {
    "https://httpbin.org": "block",
    "https://example.org": "allow",
  });
});

test("the agent's history access set in another window survives this one's changes, and chats stay", () => {
  patchAgentChat("chat-1", { tabId: "tab-1" });
  useBrowserAgentStore.getState().setShowCursor(true);
  otherWindowWrites("unsloth_browser_agent", (state) => ({ ...state, historyAccess: "never" }));
  useBrowserAgentStore.getState().setShowCursor(false);
  assert.equal(stored("unsloth_browser_agent").historyAccess, "never");
  assert.equal(useBrowserAgentStore.getState().historyAccess, "never");
  assert.equal(useBrowserAgentStore.getState().chats["chat-1"]?.tabId, "tab-1");
});

test("clearing or removing history is saved at once, so another window's queued visit cannot bring it back", () => {
  const key = "unsloth_browser_history:b1";
  const urls = () => (stored(key).history as { url: string }[]).map((visit) => visit.url).sort();
  useBrowserHistoryStore.getState().recordVisit("https://example.com/", "Example");
  useBrowserHistoryStore.getState().recordVisit("https://example.org/", "Org");
  flush();
  useBrowserHistoryStore.getState().recordVisit("https://example.net/", "Net");
  assert.deepEqual(urls(), ["https://example.com/", "https://example.org/"]);
  const org = useBrowserHistoryStore.getState().history.find((visit) => visit.url === "https://example.org/");
  useBrowserHistoryStore.getState().removeVisit(org!.id);
  assert.deepEqual(urls(), ["https://example.com/", "https://example.net/"]);
  useBrowserHistoryStore.getState().clearHistory();
  assert.deepEqual(urls(), []);
});

test("a finished download is saved at once, so another window's history write does not drop it", () => {
  const key = "unsloth_browser_history:b1";
  const id = useBrowserHistoryStore.getState().recordDownload(
    { name: "report.pdf", url: "https://example.com/report.pdf", contentType: "application/pdf", size: 10 },
    false,
  );
  otherWindowWrites(key, (state) => ({ ...state, history: [] }));
  assert.ok(useBrowserHistoryStore.getState().downloads.some((item) => item.id === id));
  assert.ok((stored(key).downloads as { id: string }[]).some((item) => item.id === id));
});
