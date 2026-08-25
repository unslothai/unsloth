// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reserving one of the three Xet notices.
//
// The count and limit moved to the server, which does the read and write in one
// transaction, so the Web Locks race this file used to cover is gone and the
// concurrency case lives in test_xet_notice_settings.py.
//
// What is left is the client half: send the legacy count once, and fail CLOSED. A
// local fallback on error would restore the resetting bug on any flaky request.

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

// Stub global fetch, not the authFetch export: ES module namespaces are frozen. This
// also exercises the real header and URL handling. The Tauri retry it wraps only
// engages under Tauri, so a thrown error surfaces immediately.
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
  // Fail closed: a local fallback would reinstate the resetting bug on every failure.
  reset();
  respond = async () => new Response("{}", { status: 500 });
  assert.equal((await reserveXetNoticeFromServer()).granted, false);

  reset();
  respond = async () => {
    throw new Error("network down");
  };
  assert.equal((await reserveXetNoticeFromServer()).granted, false);

  // An older backend has no such route, so the body is not what we expect.
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

  // Second call does not resend it: the server is authoritative from here.
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[1].body, { seen_hint: 0 });
});

test("a legacy count survives a failed reservation", async () => {
  // Marking it migrated on the way out dropped the floor whenever the POST failed:
  // every later request sent 0, so a user who had spent their three got three more.
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
  // A proxy or an older backend can answer this unknown route with a 200 and some
  // other JSON. Treating that as proof the hint was stored drops the floor, and the
  // three notices come back on the next upgrade.
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
