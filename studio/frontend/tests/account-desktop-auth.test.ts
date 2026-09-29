// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type AutoAuth = {
  tauriAutoAuth(options?: { force?: boolean }): Promise<boolean>;
  isTauriLoginRequired(): boolean;
  getTauriAuthFailure(): string | null;
};

type Guards = {
  requireAuth(): Promise<void>;
  requireGuest(): Promise<void>;
  requirePasswordChangeFlow(): Promise<void>;
};

function fixture(t: { after(fn: () => void): void }, tauri = true) {
  const originalWindow = globalThis.window;
  const originalFetch = globalThis.fetch;
  const events: string[] = [];
  Object.assign(globalThis, {
    window: { dispatchEvent: (event: Event) => events.push(event.type) },
  });
  t.after(() => {
    Object.assign(globalThis, { window: originalWindow, fetch: originalFetch });
  });
  let access: string | null = null;
  let refresh: string | null = null;
  let passwordChange = false;
  let response: unknown = { access_token: "owner-access", refresh_token: "owner-refresh" };
  let invokeCalls = 0;
  let refreshCalls = 0;
  let refreshSucceeds = false;
  let statusCalls = 0;
  let status: {
    initialized: boolean;
    requires_password_change: boolean;
    login_mode: string;
    full_access?: boolean;
  } = { initialized: true, requires_password_change: false, login_mode: "single" };
  const navigations: unknown[] = [];
  const session = {
    hasAuthToken: () => !!access,
    hasRefreshToken: () => !!refresh,
    mustChangePassword: () => passwordChange,
    setMustChangePassword: (value: boolean) => { passwordChange = value; },
    storeAuthTokens: (a: string, r: string) => { access = a; refresh = r; },
    clearAuthTokens: () => { access = null; refresh = null; passwordChange = false; },
    getPostAuthRoute: () => tauri ? "/chat" : passwordChange ? "/change-password" : "/chat",
    refreshSession: async () => {
      refreshCalls++;
      if (!refreshSucceeds) return false;
      access = "refreshed-access";
      refresh = "refreshed-refresh";
      return true;
    },
  };
  const apiBase = { isTauri: tauri, apiUrl: (path: string) => path };
  const transitions: { account: unknown; route: string }[] = [];
  const loginModes: unknown[] = [];
  const auth = loadWithStubs<AutoAuth>(new URL("../src/features/auth/tauri-auto-auth.ts", import.meta.url), {
    "@/lib/api-base": apiBase,
    "./session": session,
    "./api": session,
    "./login-client": {
      setLoginMode: (mode: string, fullAccess?: boolean) => {
        loginModes.push([mode, fullAccess]);
      },
    },
    "@/lib/account-transition": {
      OWNER_BROWSER_ACCOUNT: "unsloth",
      transitionBrowserAccount: async (
        account: unknown,
        route: string,
        commitSession: () => void,
      ) => {
        transitions.push({ account, route });
        commitSession();
        return false;
      },
    },
    "@tauri-apps/api/core": {
      invoke: async (command: string) => {
        assert.equal(command, "desktop_auth");
        invokeCalls++;
        if (response instanceof Error) throw response;
        return response;
      },
    },
    "@/app/router": { router: { navigate: async (options: unknown) => { navigations.push(options); } } },
  });
  const guards = loadWithStubs<Guards>(new URL("../src/app/auth-guards.ts", import.meta.url), {
    "@tanstack/react-router": { redirect: (options: unknown) => options },
    "@/lib/api-base": apiBase,
    "@/features/auth": session,
    "@/features/auth/tauri-auto-auth": auth,
    "@/features/auth/login-client": {
      setLoginMode: (mode: string, fullAccess?: boolean) => {
        loginModes.push([mode, fullAccess]);
      },
    },
  });
  globalThis.fetch = async () => {
    statusCalls++;
    return new Response(JSON.stringify(status), { status: 200 });
  };
  return {
    auth, guards, session, events, navigations, loginModes, transitions,
    setResponse: (value: unknown) => { response = value; },
    setStatus: (value: typeof status) => { status = value; },
    setRefreshSucceeds: (value: boolean) => { refreshSucceeds = value; },
    get access() { return access; },
    get refresh() { return refresh; },
    get invokeCalls() { return invokeCalls; },
    get refreshCalls() { return refreshCalls; },
    get statusCalls() { return statusCalls; },
  };
}

const multi = { login_required: true, login_mode: "multi" };

test("single-account desktop keeps token storage, password bypass and coalescing", async (t) => {
  const f = fixture(t);
  f.session.setMustChangePassword(true);
  assert.deepEqual(await Promise.all([f.auth.tauriAutoAuth(), f.auth.tauriAutoAuth()]), [true, true]);
  assert.equal(f.invokeCalls, 1);
  assert.equal(f.access, "owner-access");
  assert.equal(f.refresh, "owner-refresh");
  assert.equal(f.session.mustChangePassword(), false);
  assert.equal(f.auth.isTauriLoginRequired(), false);
  assert.deepEqual(f.navigations, []);
});

test("single-account cached session stays local and startup guards keep their routes", async (t) => {
  const f = fixture(t);
  f.session.storeAuthTokens("existing", "refresh");
  assert.equal(await f.auth.tauriAutoAuth(), true);
  await f.guards.requireAuth();
  await assert.rejects(f.guards.requireGuest(), { to: "/chat" });
  await assert.rejects(f.guards.requirePasswordChangeFlow(), { to: "/chat" });
  assert.equal(f.invokeCalls, 0);
  assert.equal(f.refreshCalls, 0);
  assert.equal(f.statusCalls, 0);
});

test("forced multi-account startup opens login and releases the startup screen without tokens", async (t) => {
  const f = fixture(t);
  f.session.storeAuthTokens("old-owner", "old-refresh");
  f.setResponse(multi);
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
  assert.equal(f.auth.isTauriLoginRequired(), true);
  assert.equal(f.access, null);
  assert.equal(f.refresh, null);
  assert.deepEqual(f.navigations, [{ to: "/login", replace: true }]);
  assert.equal(f.auth.getTauriAuthFailure(), null);
  assert.deepEqual(f.events, []);
  await f.guards.requireGuest();
  await assert.rejects(f.guards.requireAuth(), { to: "/login" });
  await assert.rejects(f.guards.requirePasswordChangeFlow(), { to: "/login" });
});

test("ordinary multi-account API recovery never reports an authenticated session", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  assert.equal(await f.auth.tauriAutoAuth(), false);
  assert.equal(f.access, null);
  assert.equal(f.auth.getTauriAuthFailure(), null);
});

test("an API retry sharing the forced startup probe still requires a session", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  const startup = f.auth.tauriAutoAuth({ force: true });
  const recovery = f.auth.tauriAutoAuth();
  assert.deepEqual(await Promise.all([startup, recovery]), [true, false]);
  assert.equal(f.invokeCalls, 1);
});

test("returning to one account restores desktop auto-login and guard shortcuts", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  await f.auth.tauriAutoAuth({ force: true });
  f.setResponse({ access_token: "restored-owner", refresh_token: "restored-refresh" });
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
  assert.equal(f.auth.isTauriLoginRequired(), false);
  assert.equal(f.access, "restored-owner");
  await f.guards.requireAuth();
  await assert.rejects(f.guards.requireGuest(), { to: "/chat" });
  assert.equal(f.statusCalls, 0);
});

for (const passwordChange of [false, true]) {
  test(`a desktop account switch keeps the managed session it just created, password change ${passwordChange}`, async (t) => {
    const f = fixture(t);
    f.setResponse(multi);
    f.session.storeAuthTokens("alice-access", "alice-refresh");
    f.session.setMustChangePassword(passwordChange);
    f.setRefreshSucceeds(true);
    assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
    assert.equal(f.access, "refreshed-access");
    assert.equal(f.refreshCalls, 1);
    assert.deepEqual(
      f.navigations,
      passwordChange ? [{ to: "/change-password", replace: true }] : [],
    );
    if (!passwordChange) await f.guards.requireAuth();
  });
}

test("owner tokens after the last managed account is deleted clear that account's browser data", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  await f.auth.tauriAutoAuth({ force: true });
  f.setResponse({ access_token: "owner-access", refresh_token: "owner-refresh" });
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
  assert.deepEqual(f.transitions, [{ account: "unsloth", route: "/chat" }]);
  assert.deepEqual(f.loginModes, [["single", undefined]]);
  assert.equal(f.access, "owner-access");
});

test("a desktop probe with no usable refresh token still opens login", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  f.session.storeAuthTokens("stale-access", "stale-refresh");
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
  assert.equal(f.refreshCalls, 1);
  assert.equal(f.access, null);
  assert.deepEqual(f.navigations, [{ to: "/login", replace: true }]);
});

test("a guard status read publishes the account policy it just learned", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  await f.auth.tauriAutoAuth({ force: true });
  f.session.storeAuthTokens("alice-access", "alice-refresh");
  f.setStatus({
    initialized: true,
    requires_password_change: false,
    login_mode: "multi",
    full_access: false,
  });
  await f.guards.requireAuth();
  assert.deepEqual(f.loginModes, [["multi", false]]);
});

test("a managed session stays signed in without another desktop exchange", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  await f.auth.tauriAutoAuth({ force: true });
  f.session.storeAuthTokens("alice-access", "alice-refresh");
  assert.equal(await f.auth.tauriAutoAuth(), true);
  assert.equal(f.access, "alice-access");
  assert.equal(f.invokeCalls, 1);
});

for (const tauri of [true, false]) {
  for (const requiresChange of [true, false]) {
    test(`${tauri ? "desktop" : "browser"} multi-account guards use session password flag ${requiresChange}`, async (t) => {
      const f = fixture(t, tauri);
      if (tauri) {
        f.setResponse(multi);
        await f.auth.tauriAutoAuth({ force: true });
      }
      f.session.storeAuthTokens("alice-access", "alice-refresh");
      f.session.setMustChangePassword(requiresChange);
      f.setStatus({ initialized: true, requires_password_change: !requiresChange, login_mode: "multi" });
      if (requiresChange) {
        await assert.rejects(f.guards.requireAuth(), { to: "/change-password" });
        await f.guards.requirePasswordChangeFlow();
      } else {
        await f.guards.requireAuth();
        await assert.rejects(f.guards.requireGuest(), { to: "/chat" });
      }
      assert.equal(f.session.mustChangePassword(), requiresChange);
    });
  }
}

test("backend-not-ready retains retry behavior and does not open login", async (t) => {
  const f = fixture(t);
  f.setResponse(new Error("Backend is not ready"));
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), false);
  assert.equal(f.auth.getTauriAuthFailure(), null);
  assert.deepEqual(f.navigations, []);
});

test("desktop secret failures remain failures", async (t) => {
  const f = fixture(t);
  f.setResponse(new Error("Desktop authentication failed"));
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), false);
  assert.equal(f.auth.getTauriAuthFailure(), "Desktop authentication failed");
  assert.deepEqual(f.events, ["tauri-auth-failed"]);
  assert.deepEqual(f.navigations, []);
});

test("browser mode never invokes the desktop shell", async (t) => {
  const f = fixture(t, false);
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), false);
  assert.equal(f.invokeCalls, 0);
});
