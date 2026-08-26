// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The History grid re-reads /api/health on a timer to keep the Copy preview link base
// live, which turned two latent races in the platform store into recurring ones: an
// unauthenticated reply nulling a tunnel that is still up, and two forced reads landing
// out of order. Either leaves the button copying a link only this machine can open.

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
  value: {
    platform: "Linux x86_64",
    userAgent: "Mozilla/5.0 (X11; Linux x86_64)",
  },
});

const TUNNEL = "https://live.trycloudflare.com";

// Authed: device_type only reaches a caller the backend recognises, and the tunnel
// fields ride with it.
const AUTHED = {
  device_type: "linux",
  chat_only: false,
  chat_only_reason: null,
  version: "2026.1.1",
  cloudflare_url: TUNNEL,
  server_url: "http://10.0.0.4:8000",
  secure: true,
};
// What the same endpoint answers once the token expires: HTTP 200, no authed fields.
const UNAUTHED = { chat_only: false };
// Authed, but sent while the hardware snapshot is still provisional: health_check ships
// the tunnel fields before it knows a device_type. Here the tunnel has been stopped.
const AUTHED_DETECTING = {
  chat_only: false,
  hardware_detecting: true,
  version: "2026.1.1",
  cloudflare_url: null,
  server_url: "http://10.0.0.4:8000",
  secure: false,
};

let next: () => Promise<Record<string, unknown>> = async () => AUTHED;
globalThis.fetch = (async () => {
  const body = await next();
  const res = { ok: true, json: async () => ({ ...body }), clone: () => res };
  return res;
}) as unknown as typeof fetch;

const { fetchDeviceType, usePlatformStore } = await import(
  "../src/config/env.ts"
);

test("an expired token does not null a tunnel that is still up", async () => {
  await fetchDeviceType({ force: true });
  assert.equal(
    usePlatformStore.getState().cloudflareUrl,
    TUNNEL,
    "nothing was stored",
  );

  next = async () => UNAUTHED;
  await fetchDeviceType({ force: true });

  const after = usePlatformStore.getState();
  assert.equal(
    after.cloudflareUrl,
    TUNNEL,
    "the poll nulled the tunnel URL, so Copy preview link falls back to this origin",
  );
  assert.equal(after.serverUrl, "http://10.0.0.4:8000");
  assert.equal(after.secure, true);
});

test("a superseded forced read does not restore what a later one replaced", async () => {
  next = async () => AUTHED;
  await fetchDeviceType({ force: true });

  // The stalled read: it answers with the URL that was live when it was sent.
  let releaseStalled = () => {};
  next = async () => {
    await new Promise<void>((resolve) => {
      releaseStalled = resolve;
    });
    return AUTHED;
  };
  const stalled = fetchDeviceType({ force: true });
  await new Promise((resolve) => setTimeout(resolve, 0));

  // The tunnel is restarted, and the replacement read picks up its new hostname.
  const restarted = {
    ...AUTHED,
    cloudflare_url: "https://restarted.trycloudflare.com",
  };
  next = async () => restarted;
  await fetchDeviceType({ force: true });
  assert.equal(
    usePlatformStore.getState().cloudflareUrl,
    "https://restarted.trycloudflare.com",
  );

  releaseStalled();
  await stalled;
  assert.equal(
    usePlatformStore.getState().cloudflareUrl,
    "https://restarted.trycloudflare.com",
    "the abandoned read wrote its stale tunnel URL over the live one",
  );
});

test("a stopped tunnel is cleared even before detection settles", async () => {
  next = async () => AUTHED;
  await fetchDeviceType({ force: true });
  assert.equal(usePlatformStore.getState().cloudflareUrl, TUNNEL);

  next = async () => AUTHED_DETECTING;
  await fetchDeviceType({ force: true });

  assert.equal(
    usePlatformStore.getState().cloudflareUrl,
    null,
    "an authed null is a tunnel that stopped, not a field the reply left out",
  );
  assert.equal(
    usePlatformStore.getState().deviceType,
    "linux",
    "the provisional reply still must not relabel the host",
  );
});

test("a failed read keeps the platform the backend measured", async () => {
  next = async () => AUTHED;
  await fetchDeviceType({ force: true });
  const measured = usePlatformStore.getState();
  assert.equal(measured.deviceType, "linux");
  assert.equal(measured.fetched, true);

  next = async () => {
    throw new Error("network");
  };
  await fetchDeviceType({ force: true });

  const after = usePlatformStore.getState();
  assert.equal(
    after.deviceType,
    "linux",
    "a dropped poll relabelled the host from the browser it is viewed in",
  );
  assert.equal(
    after.fetched,
    true,
    "a dropped poll threw away the measurement",
  );
});
