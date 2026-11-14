// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { beforeEach, test } from "node:test";

const data = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
const flushes: (() => void)[] = [];
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: { localStorage, location: { protocol: "http:" }, addEventListener: (type: string, flush: () => void) => type === "pagehide" && flushes.push(flush) },
});

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserHistoryStore } = await import("../src/features/browser/history-store.ts");
const { useBrowserPrefsStore } = await import("../src/features/browser/prefs-store.ts");
const { HistoryFileError, importHistoryFile, parseHistoryExport } = await import("../src/features/browser/history-import.ts");

const DAY_MS = 24 * 60 * 60 * 1000;
const now = Date.now();
const usec = (ms: number) => ms * 1000 + 321;
const fileOf = (value: unknown) => new File([JSON.stringify(value)], "History.json", { type: "application/json" });
const visit = (url: string, ms: number, title = "") => ({ url, title, time_usec: usec(ms) });

// developer.apple.com/documentation/safariservices/importing-data-exported-from-safari
const safari = {
  metadata: { browser_name: "Safari", browser_version: "18.2", data_type: "history", export_time_usec: usec(now), schema_version: 1 },
  history: [
    { url: "https://example.com/a", title: "  A page  ", time_usec: usec(now - 1000), visits_count: 3, latest_visit_was_http_get: true },
    { url: "http://example.org/", time_usec: usec(now - 2000), visits_count: 1, destination_url: "https://example.org/", destination_time_usec: usec(now - 1990) },
  ],
};
// developers.google.com/data-portability/schema-reference/chrome
const takeout = {
  "Browser History": [
    { favicon_url: "https://example.com/favicon.ico", page_transition: "LINK", title: "Chrome page", url: "https://example.com/c", client_id: "haZcHX/Q4yroJPDOP3OmDw==", time_usec: usec(now - 3000) },
  ],
};

beforeEach(() => {
  useBrowserHistoryStore.setState({ history: [], icons: {} });
  useBrowserPrefsStore.setState({ saveHistory: true, historyRetentionDays: 0 });
});

test("a Safari export gives its visits in milliseconds, with a missing title left empty", () => {
  assert.deepEqual(parseHistoryExport(JSON.stringify(safari)), [
    { url: "https://example.com/a", title: "A page", visitedAt: now - 1000 },
    { url: "http://example.org/", title: "", visitedAt: now - 2000 },
  ]);
});

test("a Chrome Takeout export gives its visits, and its encrypted notice gives none", () => {
  assert.deepEqual(parseHistoryExport(JSON.stringify(takeout)), [{ url: "https://example.com/c", title: "Chrome page", visitedAt: now - 3000 }]);
  assert.deepEqual(parseHistoryExport(JSON.stringify({ "Browser History": ["Your data is encrypted and cannot be exported"] })), []);
});

test("addresses that aren't web pages, don't parse or are too long are dropped, and titles are bounded", () => {
  const parsed = parseHistoryExport(
    JSON.stringify({
      history: [
        visit("chrome://settings/", now - 1),
        visit("file:///etc/passwd", now - 2),
        visit("javascript:alert(1)", now - 3),
        visit("https://", now - 4),
        visit(`https://example.com/${"q".repeat(3000)}`, now - 5),
        { url: 5, time_usec: usec(now - 6) },
        visit("https://example.com/ok", now - 7, "t".repeat(500)),
      ],
    }),
  );
  assert.deepEqual(parsed.map((item) => item.url), ["https://example.com/ok"]);
  assert.equal(parsed[0]?.title.length, 200);
});

test("visits in the future, before 1995, or without a numeric time are dropped", () => {
  const parsed = parseHistoryExport(
    JSON.stringify({
      "Browser History": [
        visit("https://example.com/future", now + DAY_MS),
        visit("https://example.com/old", Date.UTC(1994, 11, 31)),
        { url: "https://example.com/zero", time_usec: 0 },
        { url: "https://example.com/seconds", time_usec: Math.floor(now / 1000) },
        { url: "https://example.com/string", time_usec: String(usec(now - 1)) },
        { url: "https://example.com/none" },
        visit("https://example.com/1995", Date.UTC(1995, 0, 2)),
      ],
    }),
  );
  assert.deepEqual(parsed.map((item) => item.url), ["https://example.com/1995"]);
});

test("a file that isn't a history export is rejected with the typed error", async () => {
  for (const text of ["not json", "[]", "null", "{}", '{"history":{}}', '{"bookmarks":[]}', '{"metadata":{"data_type":"extensions"},"extensions":[]}']) {
    assert.throws(() => parseHistoryExport(text), HistoryFileError, text);
  }
  await assert.rejects(importHistoryFile(fileOf({ roots: {} })), HistoryFileError);
  await assert.rejects(importHistoryFile({ size: 101 * 1024 * 1024, text: async () => "{}" } as File), HistoryFileError);
});

test("an import merges newest first under the account, and importing the file again adds nothing", async () => {
  useBrowserHistoryStore.getState().recordVisit("https://example.com/here", "Here");
  assert.deepEqual(await importHistoryFile(fileOf(safari)), { added: 2, leftOut: 0 });
  assert.deepEqual(await importHistoryFile(fileOf(takeout)), { added: 1, leftOut: 0 });
  assert.deepEqual(useBrowserHistoryStore.getState().history.map((item) => item.url), [
    "https://example.com/here",
    "https://example.com/a",
    "http://example.org/",
    "https://example.com/c",
  ]);
  assert.deepEqual(await importHistoryFile(fileOf(safari)), { added: 0, leftOut: 0 });
  const repeated = { history: [visit("https://example.com/a", now - 1000), visit("https://example.com/b", now - 50), visit("https://example.com/b", now - 50), visit("https://example.com/a", now - 60)] };
  assert.deepEqual(await importHistoryFile(fileOf(repeated)), { added: 2, leftOut: 0 });
  assert.equal(new Set(useBrowserHistoryStore.getState().history.map((item) => item.id)).size, 6);
  for (const flush of flushes) flush();
  assert.equal(data.has("unsloth_browser_history"), false);
  assert.match(data.get("unsloth_browser_history:b1") ?? "", /example\.com\/c/);
});

test("the history limit keeps the newest visits, and counts the imported ones that didn't fit", async () => {
  useBrowserHistoryStore.getState().recordVisit("https://example.com/here", "Here");
  const many = { history: Array.from({ length: 1500 }, (_, index) => visit(`https://example.com/${index}`, now - 10_000 - index)) };
  assert.deepEqual(await importHistoryFile(fileOf(many)), { added: 999, leftOut: 501 });
  const { history } = useBrowserHistoryStore.getState();
  assert.equal(history.length, 1000);
  assert.equal(history[0]?.url, "https://example.com/here");
  assert.equal(history[999]?.url, "https://example.com/998");
  assert.deepEqual(await importHistoryFile(fileOf({ history: [visit("https://example.com/older", now - DAY_MS)] })), { added: 0, leftOut: 1 });
  assert.equal(useBrowserHistoryStore.getState().history.length, 1000);
});

test("visits past the kept period are left out", async () => {
  useBrowserPrefsStore.setState({ historyRetentionDays: 7 });
  const file = fileOf({ history: [visit("https://example.com/recent", now - DAY_MS), visit("https://example.com/stale", now - 8 * DAY_MS)] });
  assert.deepEqual(await importHistoryFile(file), { added: 1, leftOut: 1 });
  assert.deepEqual(useBrowserHistoryStore.getState().history.map((item) => item.url), ["https://example.com/recent"]);
});

test("an import is still taken with history saving off", async () => {
  useBrowserPrefsStore.setState({ saveHistory: false });
  useBrowserHistoryStore.getState().recordVisit("https://example.com/unsaved", "Unsaved");
  assert.deepEqual(await importHistoryFile(fileOf(takeout)), { added: 1, leftOut: 0 });
  assert.deepEqual(useBrowserHistoryStore.getState().history.map((item) => item.url), ["https://example.com/c"]);
});
