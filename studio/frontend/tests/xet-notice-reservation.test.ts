// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reserving one of the three Xet notices.
//
// This file used to test a Web Locks dance that serialised two tabs reading the
// same localStorage count. Both the count and the limit now live on the server
// (studio/backend/utils/xet_notice_settings.py), which does the read and the write
// in one transaction, so the race this file existed for is gone rather than
// narrowed, and the concurrency case moved to test_xet_notice_settings.py.
//
// What is left here is the client half: send the legacy count exactly once, and
// fail CLOSED on anything unexpected. Failing closed is the whole point of the
// change. The old behaviour reset every time the origin moved, which is why the
// notice never actually stopped, so a fallback to a local count on error would
// quietly restore the bug on any flaky request.

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

// Stub global fetch rather than the authFetch export: ES module namespaces are
// frozen, so a namespace property cannot be redefined. Going through the real
// authFetch also covers the header and URL handling instead of mocking it away.
// The Tauri network retry it wraps only engages under Tauri, so a thrown error
// surfaces immediately here rather than looping.
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
  // Fail closed. A local fallback here would reinstate the resetting behaviour
  // on every failed request, which is the bug this replaced.
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
  // Someone who already spent their three in localStorage must not get three
  // more the first time they run a build that counts server-side.
  reset();
  store.set(LEGACY_COUNT_KEY, "3");
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[0].body, { seen_hint: 3 });
  assert.equal(store.get(LEGACY_MIGRATED_KEY), "1");

  // Second call does not resend it: the server is authoritative from here.
  await reserveXetNoticeFromServer();
  assert.deepEqual(calls[1].body, { seen_hint: 0 });
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
