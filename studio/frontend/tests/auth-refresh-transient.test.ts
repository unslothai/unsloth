// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A busy auth.db answers /api/auth/refresh with 503 + Retry-After and keeps the refresh
// token valid. Only a 401 may end the session; anything else keeps both tokens.

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type * as AuthApi from "../src/features/auth/api.ts";
import type * as Guards from "../src/app/auth-guards.ts";

type Refresh = () => Response | Promise<Response>;

const busy = (retryAfter: string | null = "1") =>
  new Response(JSON.stringify({ detail: "busy" }), {
    status: 503,
    headers: retryAfter === null ? {} : { "Retry-After": retryAfter },
  });
const rotated = () =>
  Response.json({
    access_token: "new-access",
    refresh_token: "new-refresh",
    must_change_password: false,
  });

function fixture(t: TestContext, refreshes: Refresh[]) {
  const previousWindow = globalThis.window;
  const previousFetch = globalThis.fetch;
  const previousSetTimeout = globalThis.setTimeout;
  const previousNow = Date.now;
  const location = { pathname: "/chat", href: "" };
  const tokens = {
    access: "old-access" as string | null,
    refresh: "old-refresh" as string | null,
  };
  let now = 0;
  let clears = 0;
  let unexpected = 0;
  const delays: number[] = [];
  const refreshBodies: string[] = [];
  const apiTokens: (string | null)[] = [];
  const state = { onWait: null as null | (() => void) };
  globalThis.window = { location } as unknown as Window & typeof globalThis;
  // A fake clock: a wait advances it and resolves on the next turn, not in real time.
  Date.now = () => now;
  globalThis.setTimeout = ((fn: () => void, ms?: number) => {
    delays.push(ms ?? 0);
    const hook = state.onWait;
    state.onWait = null;
    hook?.();
    setImmediate(() => {
      now += ms ?? 0;
      fn();
    });
    return 0;
  }) as unknown as typeof setTimeout;
  globalThis.fetch = async (input, init) => {
    const path = String(input);
    if (path.endsWith("/status")) {
      return Response.json({
        login_mode: "single",
        requires_password_change: false,
        initialized: true,
      });
    }
    if (path.endsWith("/logout")) return new Response(null, { status: 200 });
    if (path.endsWith("/refresh")) {
      refreshBodies.push(String(init?.body));
      const next = refreshes.shift();
      if (!next) {
        unexpected++;
        return new Response(null, { status: 418 });
      }
      return next();
    }
    const auth = new Headers(init?.headers).get("Authorization");
    apiTokens.push(auth);
    return new Response(null, {
      status: auth === "Bearer new-access" ? 200 : 401,
    });
  };
  t.after(() => {
    globalThis.window = previousWindow;
    globalThis.fetch = previousFetch;
    globalThis.setTimeout = previousSetTimeout;
    Date.now = previousNow;
    assert.equal(unexpected, 0, "a refresh ran past its script");
  });
  const session = {
    getAuthToken: () => tokens.access,
    getRefreshToken: () => tokens.refresh,
    hasAuthToken: () => !!tokens.access,
    hasRefreshToken: () => !!tokens.refresh,
    mustChangePassword: () => false,
    setMustChangePassword: () => undefined,
    storeAuthTokens: (access: string, refresh: string) => {
      tokens.access = access;
      tokens.refresh = refresh;
    },
    clearAuthTokens: () => {
      clears++;
      tokens.access = null;
      tokens.refresh = null;
    },
  };
  const apiBase = {
    apiUrl: (path: string) => path,
    getApiPort: () => null,
    isTauri: false,
  };
  const api = loadWithStubs<typeof AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": apiBase,
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "./session": session,
    },
  );
  const settle = async () => {
    for (let i = 0; i < 5; i++) {
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
  };
  return {
    api,
    session,
    apiBase,
    tokens,
    location,
    delays,
    refreshBodies,
    apiTokens,
    state,
    settle,
    clears: () => clears,
  };
}

test("a 503 then 200 is retried and stores the rotated pair", async (t) => {
  const f = fixture(t, [() => busy("0"), () => busy("1"), rotated]);
  assert.equal(await f.api.refreshSession(), true);
  assert.equal(f.refreshBodies.length, 3);
  assert.deepEqual(f.delays, [250, 1000]);
  assert.deepEqual(f.tokens, { access: "new-access", refresh: "new-refresh" });
});

test("a persistent 503 is bounded and keeps both tokens", async (t) => {
  const f = fixture(t, [() => busy(null), () => busy("60"), () => busy()]);
  assert.equal(await f.api.refreshSessionOutcome(), "unavailable");
  assert.equal(f.refreshBodies.length, 3);
  assert.deepEqual(f.delays, [1000, 5000]);
  assert.deepEqual(f.tokens, { access: "old-access", refresh: "old-refresh" });
  assert.equal(f.clears(), 0);
  await f.settle();
  assert.equal(f.location.href, "");
});

test("retries stop once the wait would pass the budget", async (t) => {
  const f = fixture(t, [() => busy("5"), () => busy("5")]);
  assert.equal(await f.api.refreshSessionOutcome(), "unavailable");
  assert.equal(f.refreshBodies.length, 2);
  assert.deepEqual(f.delays, [5000]);
});

test("a 429 is retried like a 503", async (t) => {
  const f = fixture(t, [
    () => new Response(null, { status: 429, headers: { "Retry-After": "2" } }),
    rotated,
  ]);
  assert.equal(await f.api.refreshSessionOutcome(), "ok");
  assert.deepEqual(f.delays, [2000]);
});

test("a 500 is not retried and keeps both tokens", async (t) => {
  const f = fixture(t, [() => new Response("boom", { status: 500 })]);
  assert.equal(await f.api.refreshSessionOutcome(), "unavailable");
  assert.equal(f.refreshBodies.length, 1);
  assert.deepEqual(f.delays, []);
  assert.deepEqual(f.tokens, { access: "old-access", refresh: "old-refresh" });
  assert.equal(f.clears(), 0);
});

test("a network error keeps both tokens", async (t) => {
  const f = fixture(t, [
    () => {
      throw new TypeError("Failed to fetch");
    },
  ]);
  assert.equal(await f.api.refreshSessionOutcome(), "unavailable");
  assert.deepEqual(f.tokens, { access: "old-access", refresh: "old-refresh" });
});

test("a malformed 200 body keeps both tokens", async (t) => {
  const f = fixture(t, [() => new Response("not json", { status: 200 })]);
  assert.equal(await f.api.refreshSessionOutcome(), "unavailable");
  assert.deepEqual(f.tokens, { access: "old-access", refresh: "old-refresh" });
});

test("a 401 still clears the tokens and redirects to /login", async (t) => {
  const f = fixture(t, [() => new Response(null, { status: 401 })]);
  const response = await f.api.authFetch("/api/models");
  assert.equal(response.status, 401);
  assert.equal(f.clears() > 0, true);
  assert.deepEqual(f.tokens, { access: null, refresh: null });
  await f.settle();
  assert.equal(f.location.href, "/login");
});

test("a pair rotated by another tab mid-retry stops without replay or clear", async (t) => {
  const f = fixture(t, [() => busy("1")]);
  f.state.onWait = () => {
    f.tokens.access = "peer-access";
    f.tokens.refresh = "peer-refresh";
  };
  assert.equal(await f.api.refreshSessionOutcome(), "unavailable");
  assert.equal(f.refreshBodies.length, 1, "the old token was replayed");
  assert.deepEqual(f.tokens, { access: "peer-access", refresh: "peer-refresh" });
  assert.equal(f.clears(), 0);
});

test("a logout during the retry wait ends the refresh", async (t) => {
  const f = fixture(t, [() => busy("1")]);
  let loggedOut: Promise<void> | null = null;
  f.state.onWait = () => {
    loggedOut = f.api.logout();
  };
  assert.equal(await f.api.refreshSessionOutcome(), "rejected");
  await loggedOut;
  assert.equal(f.refreshBodies.length, 1, "the refresh outlived the logout");
  assert.deepEqual(f.tokens, { access: null, refresh: null });
});

test("authFetch hands back the original 401 while refresh is unavailable", async (t) => {
  const f = fixture(t, [() => busy("0"), () => busy("0"), () => busy("0")]);
  const response = await f.api.authFetch("/api/models");
  assert.equal(response.status, 401);
  assert.equal(f.refreshBodies.length, 3);
  assert.deepEqual(f.apiTokens, ["Bearer old-access"], "the request was replayed");
  assert.deepEqual(f.tokens, { access: "old-access", refresh: "old-refresh" });
  assert.equal(f.clears(), 0);
  await f.settle();
  assert.equal(f.location.href, "");
});

test("authFetch replays the request with the refreshed token", async (t) => {
  const f = fixture(t, [rotated]);
  const response = await f.api.authFetch("/api/models");
  assert.equal(response.status, 200);
  assert.deepEqual(f.apiTokens, ["Bearer old-access", "Bearer new-access"]);
});

function loadGuards(f: ReturnType<typeof fixture>) {
  return loadWithStubs<typeof Guards>(
    new URL("../src/app/auth-guards.ts", import.meta.url),
    {
      "@tanstack/react-router": {
        redirect: (target: { to: string }) =>
          Object.assign(new Error(`redirect ${target.to}`), { target }),
      },
      "@/lib/api-base": f.apiBase,
      "@/features/auth/tauri-auto-auth": { isTauriLoginRequired: () => true },
      "@/features/auth/login-client": { setLoginMode: () => undefined },
      "@/features/auth": {
        ...f.session,
        getPostAuthRoute: () => "/chat",
        refreshSession: f.api.refreshSession,
        refreshSessionOutcome: f.api.refreshSessionOutcome,
      },
    },
  );
}

test("the route guard keeps a session whose refresh is unavailable", async (t) => {
  const f = fixture(t, [() => busy("0"), () => busy("0"), () => busy("0")]);
  f.tokens.access = null;
  await loadGuards(f).requireAuth();
  assert.equal(f.tokens.refresh, "old-refresh");
});

test("the route guard still sends a rejected refresh to /login", async (t) => {
  const f = fixture(t, [() => new Response(null, { status: 401 })]);
  f.tokens.access = null;
  await assert.rejects(loadGuards(f).requireAuth(), /redirect \/login/);
  assert.equal(f.tokens.refresh, null);
});
