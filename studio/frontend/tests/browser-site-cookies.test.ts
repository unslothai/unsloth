// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined };
Object.assign(globalThis, { localStorage, window: { localStorage, location: { protocol: "http:" }, addEventListener: () => {} } });
register("./helpers/browser-store-resolver.mjs", import.meta.url);

const { clearSiteCookies, fetchBrowserPage, mirrorSiteCookies, requestForPage } = await import("../src/features/browser/api.ts");
const { readPageStorage, writePageStorage } = await import("../src/features/browser/page-storage.ts");

const SITE = "https://app.example.com";
const held = () => readPageStorage(SITE, "tab").cookie;

test("the frame's cookies follow what the backend holds for the site, deletions included", () => {
  mirrorSiteCookies(`${SITE}/dash`, [["csrf=/", "|tok"], ["theme=/", "|dark"]]);
  assert.deepEqual(held(), [["csrf=/", "|tok"], ["theme=/", "|dark"]]);
  mirrorSiteCookies(`${SITE}/other`, [["csrf=/", "|next"]]);
  assert.deepEqual(held(), [["csrf=/", "|next"]]);
  // An answer with no cookie list (a file, an older backend) leaves them alone.
  mirrorSiteCookies(`${SITE}/file`, undefined);
  assert.deepEqual(held(), [["csrf=/", "|next"]]);
});

test("a page's own cookie write reaches the backend before the page's next load is sent", async () => {
  const calls: string[] = [];
  let release = () => {};
  (globalThis as { __authFetch?: unknown }).__authFetch = (url: string, init: { body: string }) => {
    calls.push(`${url} ${init.body}`);
    if (url.endsWith("/cookie")) {
      return new Promise((resolve) => {
        release = () => resolve(new Response(null, { status: 204 }));
      });
    }
    return Promise.resolve(
      new Response(JSON.stringify({ url: `${SITE}/`, base: `${SITE}/`, html: "", refresh: null, cookies: [["token=/", "|abc"]] }), {
        headers: { "X-Unsloth-Browser-Kind": "html" },
      }),
    );
  };
  // The mirror's own writes are not sent back; a frame's go under the panel's site, not one the page names.
  mirrorSiteCookies(`${SITE}/`, []);
  writePageStorage(SITE, "tab", "cookie", "token=/", "|abc", {}, "https://evil.example/");
  assert.deepEqual(calls, [`/api/browser/cookie ${JSON.stringify({ page: `${SITE}/`, key: "token=/", value: "|abc" })}`]);
  const loading = fetchBrowserPage({ url: `${SITE}/` }, new AbortController().signal);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(calls.length, 1);
  release();
  await loading;
  assert.equal(calls.length, 2);
  assert.deepEqual(held(), [["token=/", "|abc"]]);
});

test("successive cookie writes reach the backend in the order the page made them", async () => {
  const sent: string[] = [];
  const pending: Array<() => void> = [];
  (globalThis as { __authFetch?: unknown }).__authFetch = (_url: string, init: { body: string }) => {
    sent.push(JSON.parse(init.body).value);
    return new Promise((resolve, reject) => {
      pending.push(() => (sent.length === 1 ? reject(new Error("offline")) : resolve(new Response(null, { status: 204 }))));
    });
  };
  writePageStorage(SITE, "tab", "cookie", "step=/", "|1", {}, `${SITE}/`);
  writePageStorage(SITE, "tab", "cookie", "step=/", null, {}, `${SITE}/`);
  assert.deepEqual(sent, ["|1"]);
  pending.shift()?.();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(sent, ["|1", null]);
  pending.shift()?.();
});

test("clearing site cookies waits for a write in flight and drops the ones queued behind it", async () => {
  const sent: string[] = [];
  const pending: Array<() => void> = [];
  (globalThis as { __authFetch?: unknown }).__authFetch = (url: string, init: { body?: string }) => {
    sent.push(url.endsWith("/clear-cookies") ? "clear" : JSON.parse(init.body ?? "{}").value);
    return new Promise((resolve) => pending.push(() => resolve(new Response(null, { status: 204 }))));
  };
  writePageStorage(SITE, "tab", "cookie", "a=/", "|1", {}, `${SITE}/`);
  writePageStorage(SITE, "tab", "cookie", "b=/", "|2", {}, `${SITE}/`);
  const cleared = clearSiteCookies();
  writePageStorage(SITE, "tab", "cookie", "c=/", "|3", {}, `${SITE}/`);
  assert.deepEqual(sent, ["|1"]);
  pending.shift()?.();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(sent, ["|1", "clear"]);
  pending.shift()?.();
  await cleared;
  await new Promise((resolve) => setTimeout(resolve, 0));
  // A write made after the clear started still goes, after it.
  assert.deepEqual(sent, ["|1", "clear", "|3"]);
  pending.shift()?.();
});

test("a cookie the page wrote while a request was out is not undone by that request's older answer", async () => {
  let release = () => {};
  (globalThis as { __authFetch?: unknown }).__authFetch = (url: string) => {
    if (url.endsWith("/cookie")) return Promise.resolve(new Response(null, { status: 204 }));
    return new Promise((resolve) => {
      release = () =>
        resolve(new Response(JSON.stringify({ status: 200, statusText: "", url: `${SITE}/api`, headers: [], body: "", cookies: [["server=/", "|1"]] })));
    });
  };
  mirrorSiteCookies(`${SITE}/`, [["old=/", "|1"]]);
  const asked = requestForPage(`${SITE}/`, { url: `${SITE}/api`, method: "GET", headers: [], body: null, redirect: "follow", integrity: "", credentials: "same-origin" }, new AbortController().signal);
  await new Promise((resolve) => setTimeout(resolve, 0));
  writePageStorage(SITE, "tab", "cookie", "token=/", "|new", {}, `${SITE}/`);
  writePageStorage(SITE, "tab", "cookie", "old=/", null, {}, `${SITE}/`);
  release();
  await asked;
  assert.deepEqual(held(), [["token=/", "|new"], ["server=/", "|1"]]);
  mirrorSiteCookies(`${SITE}/`, [["server=/", "|1"]]);
  assert.deepEqual(held(), [["server=/", "|1"]]);
});

test("an older cookie snapshot arriving after a newer one is ignored, unless the backend restarted", () => {
  const site = "https://versions.example.com";
  const cookies = () => readPageStorage(site, "tab").cookie;
  mirrorSiteCookies(`${site}/`, [["csrf=/", "|new"]], undefined, ["a", 5]);
  mirrorSiteCookies(`${site}/`, [["csrf=/", "|old"]], undefined, ["a", 4]);
  assert.deepEqual(cookies(), [["csrf=/", "|new"]]);
  mirrorSiteCookies(`${site}/`, [["csrf=/", "|restarted"]], undefined, ["b", 1]);
  assert.deepEqual(cookies(), [["csrf=/", "|restarted"]]);
});

test("a favicon is fetched without the site's cookies", async () => {
  const bodies: Array<Record<string, unknown>> = [];
  (globalThis as { __authFetch?: unknown }).__authFetch = (_url: string, init: { body: string }) => {
    bodies.push(JSON.parse(init.body));
    return Promise.resolve(new Response("x", { headers: { "Content-Type": "image/png" } }));
  };
  const { proxiedFavicon } = await import("../src/features/browser/favicon.ts");
  await proxiedFavicon("https://other.example.com/favicon.ico").catch(() => undefined);
  assert.equal(bodies[0]?.credentials, "omit");
  await fetchBrowserPage({ url: `${SITE}/` }, new AbortController().signal).catch(() => undefined);
  assert.equal(bodies[1]?.credentials, "include");
});

test("a cookie write's SameSite reaches the backend, and the mirror never sends one back", async () => {
  const sent: string[] = [];
  (globalThis as { __authFetch?: unknown }).__authFetch = (_url: string, init: { body: string }) => {
    sent.push(init.body);
    return Promise.resolve(new Response(null, { status: 204 }));
  };
  writePageStorage(SITE, "tab", "cookie", "token=/", "|secret", {}, `${SITE}/`, 1, "strict");
  writePageStorage(SITE, "tab", "cookie", "theme=/", "|dark", {}, `${SITE}/`, 2);
  mirrorSiteCookies(`${SITE}/`, [["token=/", "|rotated"]]);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(sent.map((body) => JSON.parse(body)), [
    { page: `${SITE}/`, key: "token=/", value: "|secret", same_site: "strict" },
    { page: `${SITE}/`, key: "theme=/", value: "|dark" },
  ]);
});
