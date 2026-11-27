// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// main.tsx fires an unforced fetchDeviceType at startup while __root.tsx's beforeLoad awaits its
// own, so both read /api/health side by side (and both poll a still-detecting backend).

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
const { store } = installLocalStorageFake();
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { platform: "Win32", userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
});

const AUTHED = {
  device_type: "linux",
  apple_silicon: false,
  file_manager: "files",
  chat_only: false,
  chat_only_reason: null,
  version: "2026.1.1",
};
const UNAUTHED = { chat_only: false };
const DETECTING = { chat_only: true, hardware_detecting: true, version: "2026.1.1" };

let calls: Array<string | null> = [];
let replies: Array<Record<string, unknown> | null> = [];
let fallback: Record<string, unknown> | null = UNAUTHED;
globalThis.fetch = (async (_url: string, init?: { headers?: Record<string, string> }) => {
  calls.push(init?.headers?.Authorization ?? null);
  await new Promise((resolve) => setTimeout(resolve, 5));
  const body = replies.length ? replies.shift()! : fallback;
  if (body === null) throw new TypeError("Failed to fetch");
  const res = { ok: true, json: async () => ({ ...body }), clone: () => res };
  return res;
}) as unknown as typeof fetch;

const { fetchDeviceType, forgetMeasuredVerdict, usePlatformStore } = await import(
  "../src/config/env.ts"
);
const { resetApiBase, setApiBase } = await import("../src/lib/api-base.ts");

function reset(token: string | null) {
  calls = [];
  replies = [];
  fallback = UNAUTHED;
  if (token === null) store.delete("unsloth_auth_token");
  else store.set("unsloth_auth_token", token);
  usePlatformStore.setState({ fetched: false, verdictStale: false, detectionDeferred: false });
}

test("startup's two unforced callers share one /api/health read", async () => {
  reset(null);
  const [a, b] = await Promise.all([fetchDeviceType(), fetchDeviceType()]);
  assert.equal(calls.length, 1, "each concurrent caller fetched /api/health on its own");
  assert.equal(a, "windows");
  assert.equal(b, "windows");
  assert.equal(usePlatformStore.getState().fetched, false);
  // Settled: an unauthed result stays unfetched, so the next navigation reads again.
  await fetchDeviceType();
  assert.equal(calls.length, 2);
});

test("a detecting backend is polled by one loop, not one per caller", async () => {
  reset("tok");
  replies = [DETECTING, DETECTING, AUTHED];
  fallback = AUTHED;
  const [a, b] = await Promise.all([fetchDeviceType(), fetchDeviceType()]);
  assert.equal(calls.length, 3, "both callers polled the detecting backend");
  assert.equal(a, "linux");
  assert.equal(b, "linux");
  const s = usePlatformStore.getState();
  assert.equal(s.fetched, true);
  assert.equal(s.capabilitiesUnknown(), false);
  assert.equal(s.isChatOnly(), false);
});

test("a forced refresh never joins an unforced read", async () => {
  reset(null);
  const p1 = fetchDeviceType();
  const p2 = fetchDeviceType({ force: true });
  await Promise.all([p1, p2]);
  assert.equal(calls.length, 2);
});

test("a different token does not join the read in flight", async () => {
  reset(null);
  fallback = UNAUTHED;
  const p1 = fetchDeviceType();
  store.set("unsloth_auth_token", "tok2");
  replies = [UNAUTHED, AUTHED];
  const p2 = fetchDeviceType();
  const [, b] = await Promise.all([p1, p2]);
  assert.deepEqual(calls, [null, "Bearer tok2"]);
  assert.equal(b, "linux");
});

test("an unforced read never joins a forced refresh in flight", async () => {
  reset(null);
  const p1 = fetchDeviceType({ force: true });
  const p2 = fetchDeviceType();
  await Promise.all([p1, p2]);
  assert.equal(calls.length, 2);
});

test("a read sent to another backend URL is not shared (desktop port change)", async () => {
  reset(null);
  const urls: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = ((url: string, init?: unknown) => {
    urls.push(url);
    return realFetch(url, init as RequestInit);
  }) as typeof fetch;
  try {
    setApiBase(8901);
    const p1 = fetchDeviceType();
    setApiBase(8902);
    const p2 = fetchDeviceType();
    await Promise.all([p1, p2]);
  } finally {
    globalThis.fetch = realFetch;
    resetApiBase();
  }
  assert.deepEqual(urls, ["http://127.0.0.1:8901/api/health", "http://127.0.0.1:8902/api/health"]);
});

test("a read started before a backend restart is not shared after it", async () => {
  reset(null);
  const p1 = fetchDeviceType();
  forgetMeasuredVerdict();
  const p2 = fetchDeviceType();
  await Promise.all([p1, p2]);
  assert.equal(calls.length, 2);
});

test("a failed read gives every joined caller the local fallback", async () => {
  reset(null);
  fallback = null;
  const [a, b] = await Promise.all([fetchDeviceType(), fetchDeviceType()]);
  assert.equal(a, "windows");
  assert.equal(b, "windows");
  assert.equal(usePlatformStore.getState().fetched, false);
  fallback = UNAUTHED;
  await fetchDeviceType();
  assert.equal(calls.length <= 2, true);
});
