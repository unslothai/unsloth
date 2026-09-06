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
  const api = loadWithStubs<typeof AccountsApi>(new URL("../src/features/settings/api/accounts.ts", import.meta.url), {
    "@/features/auth/api": { authFetch: async (path: string, init?: RequestInit) => {
      requests.push({ path, init });
      return response();
    } },
    "@/features/auth/login-client": { fetchAuthStatus: async () => { refreshed++; } },
    "@/lib/account-transition": { normalizeAccountUsername },
  });
  return { api, requests, refreshed: () => refreshed };
}

test("lists accounts through authenticated fetch", async () => {
  const accounts = [{ account_id: "owner", username: "unsloth", role: "owner", is_active: true }];
  const c = client(() => Response.json({ accounts }));
  assert.deepEqual(await c.api.fetchAccounts(), accounts);
  assert.deepEqual(c.requests, [{ path: "/api/accounts", init: undefined }]);
});

test("creation sends normalized username and returns one-time code and expiry", async () => {
  const setup = { username: "alice", setup_code: "one-time-code", expires_at: "2026-09-06T13:00:00Z" };
  const c = client(() => Response.json(setup));
  assert.deepEqual(await c.api.createAccount(" ALICE "), setup);
  assert.deepEqual(c.requests[0], { path: "/api/accounts", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username: "alice" }) } });
  assert.equal(c.refreshed(), 1);
});

test("regeneration encodes the username as one path segment", async () => {
  const setup = { username: "alice", setup_code: "new-code", expires_at: "2026-09-06T13:00:00Z" };
  const c = client(() => Response.json(setup));
  assert.deepEqual(await c.api.regenerateSetupCode(" A/B? "), setup);
  assert.deepEqual(c.requests[0], { path: "/api/accounts/a%2Fb%3F/setup-code", init: { method: "POST" } });
});

test("deactivate, reactivate and delete accept empty success bodies and refresh policy", async () => {
  const c = client(() => new Response(null, { status: 204 }));
  await c.api.setAccountActive("Alice", false);
  await c.api.setAccountActive("Alice", true);
  await c.api.deleteAccount("Alice");
  assert.deepEqual(c.requests, [
    { path: "/api/accounts/alice/deactivate", init: { method: "POST" } },
    { path: "/api/accounts/alice/reactivate", init: { method: "POST" } },
    { path: "/api/accounts/alice", init: { method: "DELETE" } },
  ]);
  assert.equal(c.refreshed(), 3);
});

test("authorization, conflict and validation failures are shown without reporting success", async () => {
  for (const status of [403, 404, 409, 422]) {
    const c = client(() => Response.json({ detail: "Account operation rejected" }, { status }));
    await assert.rejects(c.api.createAccount("alice"), /Account operation rejected/);
    assert.equal(c.refreshed(), 0);
  }
  const c = client(() => new Response("bad gateway", { status: 502 }));
  await assert.rejects(c.api.fetchAccounts(), /Account request failed/);
});
