// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import { parseFrameMessage } from "../src/features/browser/frame-message.ts";
import {
  clearPageStorage,
  copyTabPageStorage,
  dropTabPageStorage,
  onPageStorageClear,
  onPageStorageWrite,
  pageOrigin,
  readPageStorage,
  writePageStorage,
} from "../src/features/browser/page-storage.ts";

// The page shell's cookie and storage stand-ins (routes/browser.py), run as shipped.
const source = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8")
  .split('_PAGE_STORAGE_JS = r"""')[1]
  .split('"""')[0];

const clock = { now: Date.UTC(2026, 9, 15) };
const pageDate = { now: () => clock.now, parse: Date.parse };
class StorageEvent {
  constructor(type: string, init: Record<string, unknown>) {
    Object.assign(this, init, { type });
  }
}
const stopPages: Array<() => void> = [];
// Messages between pages and the panel, held in order when a test needs them in flight.
let deferred: Array<() => void> | null = null;
const deliver = (run: () => void) => (deferred ? deferred.push(run) : run());

function flush() {
  while (deferred?.length) {
    deferred.shift()?.();
  }
  deferred = null;
}

function reset() {
  deferred = null;
  clearPageStorage();
  for (const stop of stopPages.splice(0)) {
    stop();
  }
}

function openPage(url: string, tabId = "tab-1") {
  const origin = pageOrigin(url);
  const parent = {};
  const self = {};
  const listeners: Array<(event: { source: unknown; data: unknown }) => void> = [];
  const events: Array<{ key: string | null; newValue: string | null; storageArea: unknown; url: string }> = [];
  const window = {
    addEventListener: (type: string, listener: (event: { source: unknown; data: unknown }) => void) => {
      if (type === "message") {
        listeners.push(listener);
      }
    },
    dispatchEvent: (event: { key: string | null; newValue: string | null; storageArea: unknown; url: string }) =>
      events.push(event),
  };
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
    if (parsed?.type === "storage" && origin) {
      deliver(() => writePageStorage(origin, tabId, parsed.area, parsed.key, parsed.value, self, parsed.url));
    }
  };
  const cfg = { storage: origin ? readPageStorage(origin, tabId) : null };
  runInNewContext(source, { window, document, cfg, post, parent, Date: pageDate, StorageEvent, URL, pageUrl: url });
  const message = (data: unknown, source: unknown = parent) => {
    for (const listener of listeners) {
      listener({ source, data });
    }
  };
  // As page-frame.tsx does for every open page.
  stopPages.push(
    onPageStorageWrite(({ origin: site, area, key, value, requested, from, url: writer }) => {
      const own = from === self;
      if (site === origin && area !== "session" && !(own && value !== requested)) {
        const command = { command: "storageSync", area, key, value, url: writer ?? null, own };
        deliver(() => message({ type: "unsloth:browser-command", ...command }));
      }
    }),
  );
  return {
    window: window as unknown as Record<string, Storage>,
    document: document as { cookie: string },
    message,
    events,
  };
}

test("a cookie set on one page is there on the same site's next page", () => {
  reset();
  const first = openPage("https://en.wikipedia.org/wiki/Cat");
  first.document.cookie = "enwikimwclientpreferences=skin-theme-clientpref-night; path=/; max-age=2592000; secure";
  first.document.cookie = "consent=yes; path=/";
  assert.equal(first.document.cookie, "enwikimwclientpreferences=skin-theme-clientpref-night; consent=yes");
  const next = openPage("https://en.wikipedia.org/wiki/Dog");
  assert.equal(next.document.cookie, "enwikimwclientpreferences=skin-theme-clientpref-night; consent=yes");
});

test("expired cookies are removed", () => {
  reset();
  const page = openPage("https://example.com/");
  page.document.cookie = "a=1";
  page.document.cookie = "b=2";
  page.document.cookie = "a=; expires=Thu, 01 Jan 1970 00:00:00 GMT";
  page.document.cookie = "b=2; max-age=0";
  assert.equal(page.document.cookie, "");
  assert.equal(openPage("https://example.com/next").document.cookie, "");
});

test("localStorage carries over within the site, and sessionStorage within the tab", () => {
  reset();
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
  reset();
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
  reset();
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
  assert.deepEqual(from({ area: "cookie", key: null, value: "ignored" }), {
    type: "storage",
    area: "cookie",
    key: null,
    value: null,
    url: null,
  });
  assert.deepEqual(from({ area: "local", key: "a", value: null, url: "https://example.com/" }), {
    type: "storage",
    area: "local",
    key: "a",
    value: null,
    url: "https://example.com/",
  });
});

test("keys read and write as properties too, as on a real Storage", () => {
  reset();
  const first = openPage("https://example.com/");
  (first.window.localStorage as unknown as Record<string, string>).currentTheme = "dark";
  const next = openPage("https://example.com/docs");
  const storage = next.window.localStorage as unknown as Record<string, unknown>;
  assert.equal(storage.currentTheme, "dark");
  assert.equal("currentTheme" in storage, true);
  assert.deepEqual(Object.keys(storage), ["currentTheme"]);
  assert.equal(typeof storage.getItem, "function");
  assert.equal(storage.missing, undefined);
  Reflect.deleteProperty(storage, "currentTheme");
  assert.equal(openPage("https://example.com/").window.localStorage.getItem("currentTheme"), null);
});

test("a removal reaches the store for a key another tab wrote after this page loaded", () => {
  reset();
  const older = openPage("https://example.com/", "tab-1");
  const newer = openPage("https://example.com/", "tab-2");
  newer.window.localStorage.setItem("token", "t");
  newer.document.cookie = "sid=1";
  older.window.localStorage.removeItem("token");
  older.document.cookie = "sid=; max-age=0";
  const next = openPage("https://example.com/next", "tab-1");
  assert.equal(next.window.localStorage.getItem("token"), null);
  assert.equal(next.document.cookie, "");
  newer.window.localStorage.setItem("banner", "closed");
  older.window.localStorage.clear();
  assert.equal(openPage("https://example.com/", "tab-1").window.localStorage.length, 0);
});

test("clearing browsing data empties the pages that are open, too", () => {
  reset();
  const page = openPage("https://example.com/");
  page.window.localStorage.setItem("banner", "closed");
  page.document.cookie = "sid=1";
  const clear = { type: "unsloth:browser-command", command: "clearStorage" };
  page.message(clear, {});
  assert.equal(page.window.localStorage.getItem("banner"), "closed");
  const stop = onPageStorageClear(() => page.message(clear));
  clearPageStorage();
  stop();
  assert.equal(page.window.localStorage.getItem("banner"), null);
  assert.equal(page.document.cookie, "");
  page.window.localStorage.setItem("seen", "1");
  assert.deepEqual(readPageStorage("https://example.com", "tab-1"), { local: [["seen", "1"]], session: [], cookie: [] });
});

test("a cookie's Max-Age or Expires still ends it on later pages", () => {
  reset();
  const page = openPage("https://example.com/");
  page.document.cookie = "flag=1; Max-Age=60";
  page.document.cookie = `promo=shown; Expires=${new Date(clock.now + 120_000).toUTCString()}`;
  page.document.cookie = "theme=dark";
  assert.equal(openPage("https://example.com/next").document.cookie, "flag=1; promo=shown; theme=dark");
  clock.now += 61_000;
  assert.equal(page.document.cookie, "promo=shown; theme=dark");
  clock.now += 60_000;
  assert.equal(openPage("https://example.com/later").document.cookie, "theme=dark");
});

test("a write shows up at once in the site's other open pages", () => {
  reset();
  const first = openPage("https://example.com/", "tab-1");
  const second = openPage("https://example.com/reader", "tab-2");
  const other = openPage("https://other.example/", "tab-3");
  first.window.localStorage.setItem("state", '{"theme":"dark"}');
  first.document.cookie = "sid=1";
  first.window.sessionStorage.setItem("step", "2");
  assert.equal(second.window.localStorage.getItem("state"), '{"theme":"dark"}');
  assert.equal(second.document.cookie, "sid=1");
  assert.equal(second.window.sessionStorage.getItem("step"), null);
  assert.equal(other.window.localStorage.getItem("state"), null);
  assert.deepEqual(
    second.events.map((event) => [event.key, event.newValue]),
    [["state", '{"theme":"dark"}']],
  );
  assert.equal(second.events[0]?.storageArea, second.window.localStorage);
  assert.equal(second.events[0]?.url, "https://example.com/");
  first.window.localStorage.setItem("state", '{"theme":"dark"}');
  first.window.localStorage.removeItem("missing");
  assert.equal(second.events.length, 1);
  first.window.localStorage.removeItem("state");
  assert.equal(second.window.localStorage.getItem("state"), null);
});

test("a closed tab's sessionStorage is dropped", () => {
  reset();
  const page = openPage("https://example.com/", "tab-1");
  page.window.sessionStorage.setItem("step", "2");
  page.window.localStorage.setItem("banner", "closed");
  dropTabPageStorage("tab-1");
  const next = openPage("https://example.com/", "tab-1");
  assert.equal(next.window.sessionStorage.getItem("step"), null);
  assert.equal(next.window.localStorage.getItem("banner"), "closed");
  assert.deepEqual(readPageStorage("https://example.com", "tab-1").session, []);
});

test("a cookie keeps its path, as in a browser", () => {
  reset();
  const login = openPage("https://example.com/account/login");
  login.document.cookie = "mode=account";
  login.document.cookie = "theme=dark; path=/";
  login.document.cookie = "mode=root; path=/";
  assert.equal(openPage("https://example.com/").document.cookie, "theme=dark; mode=root");
  assert.equal(openPage("https://example.com/account/settings").document.cookie, "mode=account; theme=dark; mode=root");
  assert.equal(openPage("https://example.com/accounts").document.cookie, "theme=dark; mode=root");
  login.document.cookie = "mode=; max-age=0";
  assert.equal(openPage("https://example.com/account/settings").document.cookie, "theme=dark; mode=root");
});

test("a duplicated tab starts with the original's sessionStorage", () => {
  reset();
  const page = openPage("https://example.com/", "tab-1");
  page.window.sessionStorage.setItem("step", "2");
  copyTabPageStorage("tab-1", "tab-2");
  const copy = openPage("https://example.com/", "tab-2");
  assert.equal(copy.window.sessionStorage.getItem("step"), "2");
  copy.window.sessionStorage.setItem("step", "3");
  assert.equal(openPage("https://example.com/", "tab-1").window.sessionStorage.getItem("step"), "2");
});

test("racing writes from two open pages settle on the store's value", () => {
  reset();
  const first = openPage("https://example.com/", "tab-1");
  const second = openPage("https://example.com/", "tab-2");
  deferred = [];
  first.window.localStorage.setItem("k", "a");
  second.window.localStorage.setItem("k", "b");
  flush();
  assert.deepEqual(readPageStorage("https://example.com", "tab-1").local, [["k", "b"]]);
  assert.equal(first.window.localStorage.getItem("k"), "b");
  assert.equal(second.window.localStorage.getItem("k"), "b");
});

test("a write on its way during Clear browsing data doesn't come back", () => {
  reset();
  const page = openPage("https://example.com/");
  const stop = onPageStorageClear(() =>
    deliver(() => page.message({ type: "unsloth:browser-command", command: "clearStorage" })),
  );
  deferred = [];
  page.window.localStorage.setItem("banner", "closed");
  clearPageStorage();
  flush();
  stop();
  assert.deepEqual(readPageStorage("https://example.com", "tab-1").local, []);
  assert.equal(page.window.localStorage.getItem("banner"), null);
});

test("a cookie for a domain the page isn't on is refused", () => {
  reset();
  const page = openPage("https://www.example.com/");
  page.document.cookie = "a=1; Domain=other.example";
  page.document.cookie = "b=2; Domain=.example.com";
  page.document.cookie = "c=3; Domain=www.example.com";
  assert.equal(page.document.cookie, "b=2; c=3");
});
