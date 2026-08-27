// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The cold start, before any reply has carried the authed-only fields: until then
// fetchDeviceType accepts an unauthenticated body, because something has to seed the
// chat-only guard. What it must not do is let that seed order the forced reads -- an
// authenticated reply still in flight is the only one carrying a verdict or a tunnel.
//
// Its own file because that "has an authed reply been seen" state is per module load,
// and any earlier test that logs in would settle it before this one runs.

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
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { platform: "MacIntel", userAgent: "Mozilla/5.0 (Macintosh)" },
});

const TUNNEL = "https://cold.trycloudflare.com";
const AUTHED = {
  device_type: "linux",
  chat_only: false,
  chat_only_reason: null,
  version: "2026.1.1",
  cloudflare_url: TUNNEL,
  server_url: "http://10.0.0.4:8000",
  secure: true,
};
// No authed fields: what /api/health answers when the bearer is missing or rejected.
const UNAUTHED = { chat_only: false };

let next: () => Promise<Record<string, unknown>> = async () => AUTHED;
globalThis.fetch = (async () => {
  const body = await next();
  const res = { ok: true, json: async () => ({ ...body }), clone: () => res };
  return res;
}) as unknown as typeof fetch;

const { fetchDeviceType, usePlatformStore } = await import(
  "../src/config/env.ts"
);

test("an unauthenticated seed does not outrank an authenticated read in flight", async () => {
  let releaseAuthed = () => {};
  next = async () => {
    await new Promise<void>((resolve) => {
      releaseAuthed = resolve;
    });
    return AUTHED;
  };
  const authedRead = fetchDeviceType({ force: true });
  await new Promise((resolve) => setTimeout(resolve, 0));

  // The session is cleared mid-flight, so this one gets the unprivileged body.
  next = async () => UNAUTHED;
  await fetchDeviceType({ force: true });
  assert.equal(
    usePlatformStore.getState().deviceType,
    "mac",
    "the seed is the browser guess, which is the point of accepting it at all",
  );

  releaseAuthed();
  await authedRead;
  assert.equal(
    usePlatformStore.getState().cloudflareUrl,
    TUNNEL,
    "the unauthenticated seed took the ordering mark and discarded the real reply",
  );
  assert.equal(usePlatformStore.getState().deviceType, "linux");
});
