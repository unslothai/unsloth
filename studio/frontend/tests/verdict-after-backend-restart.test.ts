// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A repair replaces the backend. The new process answers /api/health provisionally while it
// detects, and a provisional reply keeps the stored verdict, which was still marked measured. A
// Chat only install that had just been upgraded kept "Train · Add" (main: a greyed Train) for
// the rest of the session, and clicking it offered the multi-GB install again. The provider now
// drops the measured verdict when a backend that already answered goes away, so the rows spin
// and the sidebar's unknown poll re-reads until the new process measures.

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

let reply: Record<string, unknown> = CHAT_ONLY;
globalThis.fetch = (async () => {
  const body = reply;
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

  // The repair finished and the shell is back on a new process.
  forgetMeasuredVerdict();
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), true);

  // It is still detecting. Without the stale flag this reply kept no_torch as measured, and
  // the Train row went on offering the full install.
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
  // The host did not change: the platform fields survive the provisional reply.
  assert.equal(s.deviceType, "windows");
  assert.equal(s.fileManager, "explorer");

  // Detection lands; this is the sidebar's unknown poll.
  reply = FULL;
  await fetchDeviceType({ force: true });
  s = usePlatformStore.getState();
  assert.equal(s.capabilitiesUnknown(), false);
  assert.equal(s.isChatOnly(), false, "the new verdict was not stored");
  assert.equal(s.chatOnlyReason, null);
});

test("only a backend that already answered marks the verdict stale", async () => {
  // First launch has nothing stored to distrust; a re-run of the effect while the same backend
  // stays up (auth landing later) must not either.
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
