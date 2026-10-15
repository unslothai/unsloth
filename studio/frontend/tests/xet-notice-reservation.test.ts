// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The count and limit live on the server (concurrency: test_xet_notice_settings.py). The client
// sends the legacy count once and fails CLOSED; a local fallback would restore the resetting bug.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store } = installLocalStorageFake();

const LEGACY_COUNT_KEY = "unsloth.studio.xetNoticeCount";
const LEGACY_MIGRATED_KEY = "unsloth.studio.xetNoticeMigrated";

interface FetchCall {
  url: string;
  body: unknown;
}

const calls: FetchCall[] = [];
let respond: () => Promise<Response> = async () =>
  new Response(JSON.stringify({ granted: true, shown: 1, limit: 3 }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

// Stub global fetch: ES module namespaces are frozen. The Tauri retry engages only under Tauri.
Object.defineProperty(globalThis, "fetch", {
  configurable: true,
  value: async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({
      url: String(input),
      body: typeof init?.body === "string" ? JSON.parse(init.body) : null,
    });
    return respond();
  },
});

registerBundlerResolver();

const { reserveXetNoticeFromServer } = await import(
  "../src/features/settings/api/xet-notice.ts"
);

function reset() {
  calls.length = 0;
  store.clear();
  respond = async () =>
    new Response(JSON.stringify({ granted: true, shown: 1, limit: 3 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
}

test("a granted reservation is reported as granted", async () => {
  reset();
  const result = await reserveXetNoticeFromServer();
  assert.equal(result.granted, true);
  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /\/api\/settings\/xet-notice\/reserve$/);
});

test("a refused reservation is reported as refused", async () => {
  reset();
  respond = async () =>
    new Response(JSON.stringify({ granted: false, shown: 3, limit: 3 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  const result = await reserveXetNoticeFromServer();
  assert.equal(result.granted, false);
});

test("an error response shows nothing rather than falling back", async () => {
  reset();
  respond = async () => new Response("{}", { status: 500 });
  assert.equal((await reserveXetNoticeFromServer()).granted, false);

  reset();
  respond = async () => {
    throw new Error("network down");
  };
  assert.equal((await reserveXetNoticeFromServer()).granted, false);

  reset();
  respond = async () =>
    new Response(JSON.stringify({ detail: "Not Found" }), { status: 200 });
  assert.equal((await reserveXetNoticeFromServer()).granted, false);
});

test("a legacy count is sent once, as a floor", async () => {
  // Someone who spent their three in localStorage must not get three more.
  reset();
  store.set(LEGACY_COUNT_KEY, "3");
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[0].body, { seen_hint: 3 });
  assert.equal(store.get(LEGACY_MIGRATED_KEY), "1");

  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[1].body, { seen_hint: 0 });
});

test("a legacy count survives a failed reservation", async () => {
  // Marking migrated before the POST succeeded dropped the floor on failure.
  reset();
  store.set(LEGACY_COUNT_KEY, "3");
  respond = async () => {
    throw new Error("network down");
  };
  await reserveXetNoticeFromServer();
  assert.equal(store.get(LEGACY_MIGRATED_KEY), undefined);

  respond = async () =>
    new Response(JSON.stringify({ granted: true, shown: 4, limit: 3 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[1].body, { seen_hint: 3 });
  assert.equal(store.get(LEGACY_MIGRATED_KEY), "1");
});

test("a 200 that is not a reservation does not end the migration", async () => {
  // A proxy or older backend can 200 with other JSON; that is not proof the hint was stored.
  reset();
  store.set(LEGACY_COUNT_KEY, "3");
  respond = async () =>
    new Response(JSON.stringify({ detail: "Not Found" }), { status: 200 });
  assert.equal((await reserveXetNoticeFromServer()).granted, false);
  assert.equal(store.get(LEGACY_MIGRATED_KEY), undefined);

  respond = async () =>
    new Response(JSON.stringify({ granted: true, shown: 4, limit: 3 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[1].body, { seen_hint: 3 });
  assert.equal(store.get(LEGACY_MIGRATED_KEY), "1");
});

test("junk or absent legacy counts migrate as zero", async () => {
  reset();
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[0].body, { seen_hint: 0 });

  reset();
  store.set(LEGACY_COUNT_KEY, "not a number");
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[0].body, { seen_hint: 0 });

  reset();
  store.set(LEGACY_COUNT_KEY, "-4");
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[0].body, { seen_hint: 0 });
});
