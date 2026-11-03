// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A provisional reply keeps the stored verdict, so after a repair replaced the backend the old
// no_torch stayed "measured" for the session and Train stayed greyed.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrcAsync,
  registerBundlerResolver,
} from "./helpers/kit.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
const { store } = installLocalStorageFake();
store.set("unsloth_auth_token", "token");
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { platform: "Win32", userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
});

const CHAT_ONLY = {
  device_type: "windows",
  apple_silicon: false,
  file_manager: "explorer",
  chat_only: true,
  chat_only_reason: "no_torch",
  version: "2026.1.1",
};
const DETECTING = { chat_only: true, hardware_detecting: true, version: "2026.1.1" };
const FULL = { ...CHAT_ONLY, chat_only: false, chat_only_reason: null };

let reply: Record<string, unknown> | null = CHAT_ONLY;
globalThis.fetch = (async () => {
  const body = reply;
  if (body === null) throw new TypeError("Failed to fetch");
  const res = { ok: true, json: async () => ({ ...body }), clone: () => res };
  return res;
}) as unknown as typeof fetch;

const { fetchDeviceType, forgetMeasuredVerdict, usePlatformStore } = await import(
  "../src/config/env.ts"
);

test("a replaced backend's verdict is unmeasured until the new one answers", async () => {
  await fetchDeviceType({ force: true });
  let s = usePlatformStore.getState();
  assert.equal(s.capabilitiesUnknown(), false);
  assert.equal(s.chatOnlyReason, "no_torch");

  forgetMeasuredVerdict();
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), true);

  reply = DETECTING;
  const realDateNow = Date.now;
  let clock = realDateNow();
  Date.now = () => (clock += 2000);
  try {
    await fetchDeviceType({ force: true });
  } finally {
    Date.now = realDateNow;
  }
  s = usePlatformStore.getState();
  assert.equal(s.capabilitiesUnknown(), true, "a provisional reply ended the wait");
  assert.equal(s.deviceType, "windows");
  assert.equal(s.fileManager, "explorer");

  reply = FULL;
  await fetchDeviceType({ force: true });
  s = usePlatformStore.getState();
  assert.equal(s.capabilitiesUnknown(), false);
  assert.equal(s.isChatOnly(), false, "the new verdict was not stored");
  assert.equal(s.chatOnlyReason, null);
});

test("a replaced backend under deferred detection does not inherit the old reason", async () => {
  // UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1: a deferred reply counts as settled.
  reply = CHAT_ONLY;
  await fetchDeviceType({ force: true });
  assert.equal(usePlatformStore.getState().chatOnlyReason, "no_torch");
  forgetMeasuredVerdict();
  reply = { ...DETECTING, hardware_detection_deferred: true };
  await fetchDeviceType({ force: true });
  const s = usePlatformStore.getState();
  assert.equal(s.capabilitiesUnknown(), false);
  assert.equal(s.isChatOnly(), true);
  assert.equal(s.chatOnlyReason, null);
  assert.equal(s.deviceType, "windows");
});

test("a failed read after replacing a deferred backend stays unmeasured", async () => {
  reply = { ...DETECTING, hardware_detection_deferred: true };
  await fetchDeviceType({ force: true });
  assert.equal(usePlatformStore.getState().detectionDeferred, true);
  forgetMeasuredVerdict();
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), true);
  // The catch path stores the browser guess (chat_only false on Windows): it must not read as settled.
  reply = null;
  await fetchDeviceType({ force: true });
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), true);
  reply = CHAT_ONLY;
});

test("only a backend that already answered marks the verdict stale", async () => {
  // Not on first launch, and not on an effect re-run while the same backend stays up.
  const src = await readSrcAsync("app/provider.tsx");
  assert.match(
    src,
    /if \(status === "running"\) servedRef\.current = true;\s*else if \(servedRef\.current\) verdictStaleRef\.current = true;/,
  );
  assert.match(
    src,
    /if \(verdictStaleRef\.current\) \{\s*verdictStaleRef\.current = false;\s*forgetMeasuredVerdict\(\);\s*\}\s*void fetchDeviceType\(\{ force: true \}\)/,
  );
});
