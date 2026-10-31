// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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
// /api/health reports device_type only to authed callers.
store.set("unsloth_auth_token", "token");
// Force a non-Mac host: node's own navigator reports process.platform.
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    platform: "Linux x86_64",
    userAgent: "Mozilla/5.0 (X11; Linux x86_64)",
  },
});

const DETECTING = { chat_only: true, hardware_detecting: true, version: "2026.1.1" };
const MEASURED = {
  device_type: "linux",
  chat_only: false,
  chat_only_reason: null,
  version: "2026.1.1",
};

let reply: Record<string, unknown> = DETECTING;
let fetches = 0;
globalThis.fetch = (async () => {
  fetches += 1;
  const body = reply;
  const res = { ok: true, json: async () => ({ ...body }), clone: () => res };
  return res;
}) as unknown as typeof fetch;

const { fetchDeviceType, usePlatformStore } = await import("../src/config/env.ts");

test("a slow non-Mac host converges out of the pending state", async () => {
  const realDateNow = Date.now;
  let clock = realDateNow();
  // Step the clock past the 5s detection window instead of waiting it out.
  Date.now = () => (clock += 2000);
  try {
    await fetchDeviceType();
  } finally {
    Date.now = realDateNow;
  }

  const stalled = usePlatformStore.getState();
  assert.equal(stalled.deviceType, "linux", "the scenario is a non-Mac host");
  assert.equal(stalled.fetched, false, "a provisional reply was stored as a measurement");
  assert.equal(stalled.detectionDeferred, false, "the reply was slow, not deferred");
  assert.equal(
    stalled.capabilitiesUnknown(),
    true,
    "nothing left unknown, so this is no longer the case the poll has to recover from",
  );
  assert.equal(
    stalled.isChatOnly(),
    false,
    "the browser seed off macOS, which is why the chat-only recovery poll never armed",
  );

  reply = MEASURED;
  const before = fetches;
  await fetchDeviceType({ force: true });
  assert.equal(fetches - before, 1, "one poll should be one re-read");

  const settled = usePlatformStore.getState();
  assert.equal(settled.fetched, true, "the measured reply was not stored");
  assert.equal(
    settled.capabilitiesUnknown(),
    false,
    "Train and Video keep spinning and /studio keeps its loading panel after the verdict arrived",
  );
  assert.equal(settled.isChatOnly(), false, "the measured GPU verdict was dropped");
  assert.equal(settled.deviceType, "linux");
});

test("a cached authoritative verdict is not re-read without force", async () => {
  const before = fetches;
  await fetchDeviceType();
  assert.equal(fetches, before, "the cache no longer short-circuits, so every route refetches");
});

test("the recovery poll runs while the verdict is unknown, on every platform", async () => {
  const src = await readSrcAsync("components/app-sidebar.tsx");
  const call = src.indexOf("void fetchDeviceType({ force: true })");
  assert.ok(call > 0, "the recovery poll left app-sidebar.tsx");
  const start = src.lastIndexOf("useEffect(() => {", call);
  const end = src.indexOf("]);", call) + 3;
  assert.ok(start > 0 && end > start, "could not read the recovery poll effect");
  const effect = src.slice(start, end);

  const guard = /if \(([^;]*)\) return;/.exec(effect);
  assert.ok(guard, "the poll never bails out, so it runs for the life of the app");
  assert.match(
    guard[1],
    /!capabilitiesUnknown/,
    "the poll still only arms on a chat-only or deferred host, so an unmeasured verdict on " +
      "Linux or Windows is never re-read",
  );
  assert.match(
    effect,
    /chatOnlyReason !== "mlx_unavailable" && !detectionDeferred/,
    "a repaired MLX install no longer re-enables Train without a reload",
  );
  const deps = /\}, \[([^\]]*)\]\);$/.exec(effect);
  assert.ok(deps, "could not read the effect's dependencies");
  assert.match(
    deps[1],
    /capabilitiesUnknown/,
    "the effect does not re-run when the verdict lands, so the interval outlives it",
  );
  assert.match(
    effect,
    /return \(\) => stopPolling\(\);/,
    "the interval is left running once the verdict is known",
  );
  assert.match(
    effect,
    /const stopPolling = \(\) => \{\s*\n\s*window\.clearInterval\(id\);/,
    "stopPolling no longer clears the interval",
  );
});

test("the poll is mounted on every route that gates on the verdict", async () => {
  const root = await readSrcAsync("app/routes/__root.tsx");
  const hidden = /const HIDDEN_NAVBAR_ROUTES = \[([^\]]*)\]/.exec(root);
  assert.ok(hidden, "could not find HIDDEN_NAVBAR_ROUTES in __root.tsx");
  assert.ok(
    !hidden[1].includes('"/studio"'),
    "/studio renders without the sidebar, so nothing re-reads the verdict it waits on",
  );
  assert.match(root, /<AppSidebar \/>/, "the component that owns the poll is not rendered");

  for (const page of [
    "../src/features/studio/studio-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const src = await readFile(new URL(page, import.meta.url), "utf8");
    assert.ok(
      !/fetchDeviceType\(/.test(src),
      `${page} re-reads the verdict itself instead of reading the store`,
    );
  }
  const studio = await readSrcAsync("features/studio/studio-page.tsx");
  assert.match(
    studio,
    /capabilitiesUnknown/,
    "the Train page no longer waits on the verdict it shares with the poll",
  );
});
