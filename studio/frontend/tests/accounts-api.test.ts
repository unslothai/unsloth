// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { normalizeAccountUsername } from "../src/lib/account-transition.ts";
import type * as AccountsApi from "../src/features/settings/api/accounts.ts";

function client(response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  let refreshed = 0;
  const api = loadWithStubs<typeof AccountsApi>(
    new URL("../src/features/settings/api/accounts.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return response();
        },
      },
      "@/features/auth/login-client": {
        fetchAuthStatus: async () => {
          refreshed++;
        },
      },
      "@/lib/account-transition": { normalizeAccountUsername },
    },
  );
  return { api, requests, refreshed: () => refreshed };
}

test("lists accounts through authenticated fetch", async () => {
  const accounts = [
    {
      account_id: "owner",
      username: "unsloth",
      role: "owner",
      is_active: true,
    },
  ];
  const c = client(() => Response.json({ accounts }));
  assert.deepEqual(await c.api.fetchAccounts(), accounts);
  assert.deepEqual(c.requests, [{ path: "/api/accounts", init: undefined }]);
});

const backendSetup = (setup_code: string) => ({
  account: {
    account_id: "alice-id",
    username: "alice",
    role: "user",
    is_active: true,
    created_at: "2026-09-06T12:00:00Z",
    setup_code_pending: true,
  },
  setup_code,
  setup_code_expires_at: "2026-09-06T13:00:00Z",
});

test("creation sends normalized username and returns one-time code and expiry", async () => {
  const setup = {
    account_id: "alice-id",
    username: "alice",
    setup_code: "one-time-code",
    expires_at: "2026-09-06T13:00:00Z",
  };
  const c = client(() => Response.json(backendSetup("one-time-code")));
  assert.deepEqual(await c.api.createAccount(" ALICE "), setup);
  assert.deepEqual(c.requests[0], {
    path: "/api/accounts",
    init: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: "alice" }),
    },
  });
  assert.equal(c.refreshed(), 1);
});

test("regeneration is keyed by the account id, encoded as one path segment", async () => {
  const setup = {
    account_id: "alice-id",
    username: "alice",
    setup_code: "new-code",
    expires_at: "2026-09-06T13:00:00Z",
  };
  const c = client(() => Response.json(backendSetup("new-code")));
  assert.deepEqual(await c.api.regenerateSetupCode("a/b?"), setup);
  assert.deepEqual(c.requests[0], {
    path: "/api/accounts/a%2Fb%3F/setup-code",
    init: { method: "POST" },
  });
});

test("deactivate, reactivate and delete accept empty success bodies and refresh policy", async () => {
  const c = client(() => new Response(null, { status: 204 }));
  await c.api.setAccountActive("alice-id", false);
  await c.api.setAccountActive("alice-id", true);
  await c.api.deleteAccount("alice-id");
  const patch = (active: boolean) => ({
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_active: active }),
  });
  assert.deepEqual(c.requests, [
    { path: "/api/accounts/alice-id", init: patch(false) },
    { path: "/api/accounts/alice-id", init: patch(true) },
    { path: "/api/accounts/alice-id", init: { method: "DELETE" } },
  ]);
  assert.equal(c.refreshed(), 3);
});

test("authorization, conflict and validation failures are shown without reporting success", async () => {
  for (const status of [403, 404, 409, 422]) {
    const c = client(() =>
      Response.json({ detail: "Account operation rejected" }, { status }),
    );
    await assert.rejects(
      c.api.createAccount("alice"),
      /Account operation rejected/,
    );
    assert.equal(c.refreshed(), 0);
  }
  const c = client(() => new Response("bad gateway", { status: 502 }));
  await assert.rejects(c.api.fetchAccounts(), /Account request failed/);
});
