// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import { parseFrameMessage } from "../src/features/browser/frame-message.ts";
import {
  clearPageStorage,
  pageOrigin,
  readPageStorage,
  writePageStorage,
} from "../src/features/browser/page-storage.ts";

// The page shell's cookie and storage stand-ins (routes/browser.py), run as shipped.
const source = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8")
  .split('_PAGE_STORAGE_JS = r"""')[1]
  .split('"""')[0];

function openPage(url: string, tabId = "tab-1") {
  const origin = pageOrigin(url);
  const window = {};
  for (const name of ["localStorage", "sessionStorage"]) {
    Object.defineProperty(window, name, {
      configurable: true,
      get: () => {
        throw new Error("SecurityError");
      },
    });
  }
  const document = {};
  const post = (message: Record<string, unknown>) => {
    const parsed = parseFrameMessage({ source: "unsloth-browser", ...message });
    if (parsed?.type === "storage" && origin) writePageStorage(origin, tabId, parsed.area, parsed.key, parsed.value);
  };
  const cfg = { storage: origin ? readPageStorage(origin, tabId) : null };
  runInNewContext(source, { window, document, cfg, post });
  return { window: window as Record<string, Storage>, document: document as { cookie: string } };
}

test("a cookie set on one page is there on the same site's next page", () => {
  clearPageStorage();
  const first = openPage("https://en.wikipedia.org/wiki/Cat");
  first.document.cookie = "enwikimwclientpreferences=skin-theme-clientpref-night; path=/; max-age=2592000; secure";
  first.document.cookie = "consent=yes";
  assert.equal(first.document.cookie, "enwikimwclientpreferences=skin-theme-clientpref-night; consent=yes");
  const next = openPage("https://en.wikipedia.org/wiki/Dog");
  assert.equal(next.document.cookie, "enwikimwclientpreferences=skin-theme-clientpref-night; consent=yes");
});

test("expired cookies are removed", () => {
  clearPageStorage();
  const page = openPage("https://example.com/");
  page.document.cookie = "a=1";
  page.document.cookie = "b=2";
  page.document.cookie = "a=; expires=Thu, 01 Jan 1970 00:00:00 GMT";
  page.document.cookie = "b=2; max-age=0";
  assert.equal(page.document.cookie, "");
  assert.equal(openPage("https://example.com/next").document.cookie, "");
});

test("localStorage carries over within the site, and sessionStorage within the tab", () => {
  clearPageStorage();
  const first = openPage("https://example.com/");
  first.window.localStorage.setItem("banner", "closed");
  first.window.sessionStorage.setItem("step", "2");
  const next = openPage("https://example.com/about");
  assert.equal(next.window.localStorage.getItem("banner"), "closed");
  assert.equal(next.window.sessionStorage.getItem("step"), "2");
  assert.equal(openPage("https://example.com/", "tab-2").window.sessionStorage.getItem("step"), null);
  next.window.localStorage.removeItem("banner");
  assert.equal(openPage("https://example.com/").window.localStorage.getItem("banner"), null);
  next.window.sessionStorage.clear();
  assert.equal(openPage("https://example.com/").window.sessionStorage.length, 0);
});

test("one site never sees another site's cookies or storage", () => {
  clearPageStorage();
  const site = openPage("https://a.example/");
  site.document.cookie = "session=secret";
  site.window.localStorage.setItem("token", "secret");
  for (const url of ["https://b.example/", "http://a.example/", "https://a.example:8443/", "https://sub.a.example/"]) {
    const other = openPage(url);
    assert.equal(other.document.cookie, "", url);
    assert.equal(other.window.localStorage.getItem("token"), null, url);
  }
  const local = openPage("about:blank");
  local.document.cookie = "x=1";
  assert.deepEqual(readPageStorage("null", "tab-1"), { local: [], session: [], cookie: [] });
});

test("values past the quota stay in their page, and the store stays bounded", () => {
  clearPageStorage();
  const page = openPage("https://example.com/");
  page.window.localStorage.setItem("small", "kept");
  page.window.localStorage.setItem("cache", "x".repeat(200 * 1024));
  assert.equal(page.window.localStorage.getItem("cache")?.length, 200 * 1024);
  const next = openPage("https://example.com/next");
  assert.equal(next.window.localStorage.getItem("small"), "kept");
  assert.equal(next.window.localStorage.getItem("cache"), null);
  next.window.localStorage.setItem("small", "x".repeat(200 * 1024));
  assert.equal(openPage("https://example.com/").window.localStorage.getItem("small"), null);
  for (let i = 0; i < 40; i++) writePageStorage(`https://site${i}.example`, "tab-1", "local", "k", "x".repeat(120_000));
  assert.deepEqual(readPageStorage("https://site0.example", "tab-1").local, []);
  assert.equal(readPageStorage("https://site39.example", "tab-1").local.length, 1);
});

test("malformed storage messages are dropped", () => {
  const from = (message: Record<string, unknown>) => parseFrameMessage({ source: "unsloth-browser", type: "storage", ...message });
  assert.equal(from({ area: "indexedDB", key: "a", value: "b" }), null);
  assert.equal(from({ area: "local", key: 1, value: "b" }), null);
  assert.equal(from({ area: "local", key: "a", value: {} }), null);
  assert.equal(from({ area: "local", key: "a", value: "x".repeat(128 * 1024 + 1) }), null);
  assert.deepEqual(from({ area: "cookie", key: null, value: "ignored" }), { type: "storage", area: "cookie", key: null, value: null });
  assert.deepEqual(from({ area: "local", key: "a", value: null }), { type: "storage", area: "local", key: "a", value: null });
});
