// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Before server-port the base is port 0: the device type read must not fetch it and must land where a refused fetch did.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
const { store } = installLocalStorageFake();
store.set("unsloth_auth_token", "token");
Object.assign((globalThis as unknown as { window: object }).window, {
  __TAURI_INTERNALS__: {},
  location: { protocol: "tauri:", href: "tauri://localhost/" },
});
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    platform: "MacIntel",
    userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)",
  },
});

const requested: string[] = [];
let reply: Record<string, unknown> | null = null;
globalThis.fetch = (async (input: string) => {
  requested.push(String(input));
  const body = reply;
  // What a WebView answers for http://127.0.0.1:0.
  if (body === null) throw new TypeError("Failed to fetch");
  const res = { ok: true, json: async () => ({ ...body }), clone: () => res };
  return res;
}) as unknown as typeof fetch;

const { getApiPort, setApiBase } = await import("../src/lib/api-base.ts");
const { fetchDeviceType, usePlatformStore } = await import(
  "../src/config/env.ts"
);

test("no port-0 request before server-port, and the refused-fetch fallback state", async () => {
  assert.equal(getApiPort(), null);
  const before = { ...usePlatformStore.getState() };
  // main.tsx and the root beforeLoad both read it at startup.
  const results = await Promise.all([fetchDeviceType(), fetchDeviceType()]);
  assert.deepEqual(results, ["mac", "mac"]);
  assert.deepEqual(requested, [], "fetched the placeholder port");
  const after = usePlatformStore.getState();
  for (const key of [
    "deviceType",
    "chatOnly",
    "fetched",
    "appleSilicon",
    "fileManager",
  ] as const) {
    assert.deepEqual(after[key], before[key], key);
  }
  assert.equal(after.fetched, false);
  assert.equal(after.isChatOnly(), true);
  assert.equal(after.capabilitiesUnknown(), true);

  setApiBase(8888);
  reply = {
    device_type: "mac",
    apple_silicon: true,
    file_manager: "finder",
    chat_only: false,
    chat_only_reason: null,
    version: "2026.1.1",
  };
  assert.equal(await fetchDeviceType({ force: true }), "mac");
  assert.deepEqual(requested, ["http://127.0.0.1:8888/api/health"]);
  const settled = usePlatformStore.getState();
  assert.equal(settled.fetched, true);
  assert.equal(settled.appleSilicon, true);
  assert.equal(settled.isChatOnly(), false);
});
