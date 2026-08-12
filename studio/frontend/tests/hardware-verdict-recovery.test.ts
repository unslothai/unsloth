// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Train and Video spin instead of graying out while the hardware verdict is unmeasured, so
// something has to end the spin. fetchDeviceType's bounded wait is spent at most once per page
// load, so a host that detects slower than it stores a provisional reply and `fetched` stays
// false. The sidebar's recovery poll used to return early unless the host was chat-only or
// deferred, which on Linux and Windows it is not (the store seeds chatOnly from the user agent),
// so nothing re-read /api/health: the rows spun and /studio held its loading panel until the
// user reloaded. A cold GPU host importing torch is squarely inside that window.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
const { store } = installLocalStorageFake();
// /api/health reports device_type to authed callers only, and an unauthenticated read never
// spends the detection window, so the slow-host case only exists for a signed-in caller.
store.set("unsloth_auth_token", "token");
// A non-Mac host whatever the runner is: node's own navigator reports process.platform.
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    platform: "Linux x86_64",
    userAgent: "Mozilla/5.0 (X11; Linux x86_64)",
  },
});

// The backend's pre-detection default: chat_only true, no device_type, still measuring.
const DETECTING = { chat_only: true, hardware_detecting: true, version: "2026.1.1" };
// What the same host answers once the torch import lands.
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
  // The window is 5s of wall clock with 200ms re-reads. Step the clock so the test does not
  // sit through it; detection is still unfinished when it closes, which is the whole case.
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

  // Detection lands after the window closed. This is the re-read the sidebar interval fires.
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
  // The poll passes force, so it must be force that re-reads: a plain call short-circuits on
  // the cached verdict, which is what keeps navigation instant.
  const before = fetches;
  await fetchDeviceType();
  assert.equal(fetches, before, "the cache no longer short-circuits, so every route refetches");
});

test("the recovery poll runs while the verdict is unknown, on every platform", async () => {
  const src = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
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
  // The MLX self-heal case still polls after a measured chat-only verdict.
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
    /return \(\) => window\.clearInterval\(id\);/,
    "the interval is left running once the verdict is known",
  );
});

test("the poll is mounted on every route that gates on the verdict", async () => {
  const root = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  const hidden = /const HIDDEN_NAVBAR_ROUTES = \[([^\]]*)\]/.exec(root);
  assert.ok(hidden, "could not find HIDDEN_NAVBAR_ROUTES in __root.tsx");
  assert.ok(
    !hidden[1].includes('"/studio"'),
    "/studio renders without the sidebar, so nothing re-reads the verdict it waits on",
  );
  assert.match(root, /<AppSidebar \/>/, "the component that owns the poll is not rendered");

  // Neither page starts a poll of its own. Video needs none at all -- its answer is
  // /api/system/hardware's, which settles detection before replying -- so it is absent from the
  // waits-on-the-verdict assertion below, but a second /api/health poll would still be wrong.
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
  const studio = await readFile(
    new URL("../src/features/studio/studio-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    studio,
    /capabilitiesUnknown/,
    "the Train page no longer waits on the verdict it shares with the poll",
  );
});
