// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Settings = {
  enabled: boolean;
  effective_enabled: boolean;
  disabled_by_env: boolean;
};

/** Load the real settings client with a controlled HTTP boundary. */
function settingsApi(
  fetch: (url: string, init?: RequestInit) => Promise<Response>,
) {
  return loadWithStubs<{
    loadApiMonitorSettings(): Promise<Settings>;
    updateApiMonitorSettings(enabled: boolean): Promise<Settings>;
  }>(new URL("../src/features/settings/api/api-monitor.ts", import.meta.url), {
    "@/features/auth": { authFetch: fetch },
    "@/lib/format-fastapi-error": {
      readFastApiError: async (response: Response) =>
        (await response.json()).detail,
    },
  });
}

test("reads and saves the server's effective recording state", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const api = settingsApi(async (url, init) => {
    calls.push({ url, init });
    return Response.json({
      enabled: true,
      effective_enabled: false,
      disabled_by_env: true,
    });
  });
  assert.deepEqual(await api.loadApiMonitorSettings(), {
    enabled: true,
    effective_enabled: false,
    disabled_by_env: true,
  });
  assert.deepEqual(await api.updateApiMonitorSettings(false), {
    enabled: true,
    effective_enabled: false,
    disabled_by_env: true,
  });
  assert.equal(calls[0].url, "/api/settings/api-monitor");
  assert.equal(calls[1].url, "/api/settings/api-monitor");
  assert.equal(calls[1].init?.method, "PUT");
  assert.equal(calls[1].init?.body, JSON.stringify({ enabled: false }));
});

test("surfaces settings read and write errors", async () => {
  const api = settingsApi(async () =>
    Response.json({ detail: "Owner required" }, { status: 403 }),
  );
  await assert.rejects(api.loadApiMonitorSettings(), /Owner required/);
  await assert.rejects(api.updateApiMonitorSettings(false), /Owner required/);
});
