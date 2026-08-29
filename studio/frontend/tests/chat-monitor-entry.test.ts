// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { afterEach } from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const {
  ApiMonitorEntryRequestError,
  getApiMonitorEntry,
  isPermanentApiMonitorEntryError,
} = await import("../src/features/chat/api/chat-monitor.ts");

test("a stale monitor id is a permanent request error", async () => {
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ detail: "API monitor entry not found" }), {
      status: 404,
      headers: { "content-type": "application/json" },
    })) as typeof fetch;

  await assert.rejects(getApiMonitorEntry("missing"), (error: unknown) => {
    assert.ok(error instanceof ApiMonitorEntryRequestError);
    assert.equal(error.status, 404);
    assert.equal(isPermanentApiMonitorEntryError(error), true);
    return true;
  });
});

test("a wedged monitor read aborts within its request budget", async () => {
  globalThis.fetch = ((_input: RequestInfo | URL, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener(
        "abort",
        () => reject(new DOMException("timed out", "AbortError")),
        { once: true },
      );
    })) as typeof fetch;

  await assert.rejects(getApiMonitorEntry("wedged", undefined, 5), {
    name: "AbortError",
  });
});
