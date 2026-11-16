// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined };
Object.assign(globalThis, { localStorage, window: { localStorage, location: { protocol: "http:" }, addEventListener: () => {} } });
register("./helpers/browser-store-resolver.mjs", import.meta.url);

const { fetchBrowserPage, mirrorSiteCookies } = await import("../src/features/browser/api.ts");
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
  // The mirror's own writes are not sent back; a frame's are, under the site the panel loaded, not one the page names.
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
  // The delete waits for the set, and a failed set doesn't hold it back.
  assert.deepEqual(sent, ["|1"]);
  pending.shift()?.();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(sent, ["|1", null]);
  pending.shift()?.();
});
