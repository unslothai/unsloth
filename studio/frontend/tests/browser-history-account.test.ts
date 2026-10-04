// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const data = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
const flushes: (() => void)[] = [];
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: { localStorage, location: { protocol: "http:" }, addEventListener: (_: string, flush: () => void) => flushes.push(flush) },
});

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserHistoryStore } = await import("../src/features/browser/history-store.ts");

test("history persists under the account that made it, so a write flushed after a switch stays there", () => {
  useBrowserHistoryStore.getState().recordVisit("https://example.com/", "Example");
  data.set("unsloth.browser-account.v1", "unsloth");
  for (const flush of flushes) flush();
  assert.equal(data.has("unsloth_browser_history"), false);
  assert.match(data.get("unsloth_browser_history:b1") ?? "", /example\.com/);
});

test("a download keeps its record but not an address too long to store", () => {
  const store = useBrowserHistoryStore.getState();
  store.recordDownload({ name: "a.zip", url: `https://example.com/${"q".repeat(5000)}`, size: 1, contentType: "x" });
  store.recordDownload({ name: "b.zip", url: "https://example.com/b.zip", size: 1, contentType: "x" });
  const [b, a] = useBrowserHistoryStore.getState().downloads;
  assert.equal(a?.url, null);
  assert.equal(b?.url, "https://example.com/b.zip");
});
