// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as SystemHooks from "../src/hooks/use-system.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const WINDOW_MS = 10_000;
const REFRESH_URL = "/api/system?refresh_memory=true";

function loadSystem() {
  const requests: { url: string; resolve: (response: Response) => void }[] = [];
  const system = loadWithStubs<typeof SystemHooks>(
    new URL("../src/hooks/use-system.ts", import.meta.url),
    {
      react: {},
      "@/config/env": { refreshDeviceTypeAfterSystemInfo: () => {} },
      "@/features/auth": {
        authFetch: (url: string) =>
          new Promise<Response>((resolve) => {
            requests.push({ url, resolve });
          }),
      },
    },
    { relativePassthrough: true },
  );
  return { system, requests };
}

const refreshed = (availableGb: number) =>
  Response.json({
    memory_refreshed: true,
    memory: { available_gb: availableGb },
  });

test("a Hub refresh is one probe per window, shared by concurrent callers", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  const { system, requests } = loadSystem();
  const first = system.refreshSystemMemory({ maxAgeMs: WINDOW_MS });
  const second = system.refreshSystemMemory({ maxAgeMs: WINDOW_MS });
  assert.equal(
    requests.length,
    1,
    "concurrent callers share the in-flight probe",
  );
  assert.equal(requests[0].url, REFRESH_URL);
  requests[0].resolve(refreshed(10));
  const landed = await first;
  assert.equal(await second, landed);
  t.mock.timers.tick(WINDOW_MS - 1);
  assert.equal(
    await system.refreshSystemMemory({ maxAgeMs: WINDOW_MS }),
    landed,
  );
  assert.equal(requests.length, 1, "inside the window the snapshot is reused");
  t.mock.timers.tick(1);
  const later = system.refreshSystemMemory({ maxAgeMs: WINDOW_MS });
  assert.equal(requests.length, 2, "after the window a new probe goes out");
  assert.equal(requests[1].url, REFRESH_URL);
  requests[1].resolve(refreshed(6));
  assert.equal((await later)?.memory.available_gb, 6);
});

test("only a refresh_memory reading starts the window; a failed probe does not", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  const { system, requests } = loadSystem();
  const plain = system.fetchSystemInfo();
  requests[0].resolve(Response.json({ memory: { available_gb: 20 } }));
  await plain;
  const failing = system.refreshSystemMemory({ maxAgeMs: WINDOW_MS });
  assert.equal(requests.length, 2, "a plain snapshot is not a fresh reading");
  assert.equal(requests[1].url, REFRESH_URL);
  requests[1].resolve(new Response(null, { status: 503 }));
  assert.equal(await failing, null);
  const retry = system.refreshSystemMemory({ maxAgeMs: WINDOW_MS });
  assert.equal(requests.length, 3, "a failed probe starts no window");
  requests[2].resolve(refreshed(9));
  assert.equal((await retry)?.memory.available_gb, 9);
});

test("the chat picker's own probe counts, so the Hub does not re-query behind it", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  const { system, requests } = loadSystem();
  const probe = system.fetchSystemInfo({ refreshMemory: true });
  requests[0].resolve(refreshed(12));
  const snapshot = await probe;
  assert.equal(
    await system.refreshSystemMemory({ maxAgeMs: WINDOW_MS }),
    snapshot,
  );
  assert.equal(requests.length, 1);
});
