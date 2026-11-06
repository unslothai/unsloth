// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "@typescript/typescript6";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type AutoAuth = {
  tauriAutoAuth(options?: { force?: boolean }): Promise<boolean>;
  isTauriLoginRequired(): boolean;
  getTauriAuthFailure(): string | null;
  markDesktopSignedOut(): Promise<void>;
  continueAsDesktopOwner(): Promise<boolean>;
};

type Guards = {
  requireAuth(): Promise<void>;
  requireGuest(): Promise<void>;
  requirePasswordChangeFlow(): Promise<void>;
};

function fixture(
  t: { after(fn: () => void): void },
  tauri = true,
  // The Rust shell's sign-out marker file, shared by every window and every launch.
  shell = { signedOut: false },
) {
  const originalWindow = globalThis.window;
  const originalFetch = globalThis.fetch;
  const events: string[] = [];
  Object.assign(globalThis, {
    window: {
      dispatchEvent: (event: Event) => events.push(event.type),
      localStorage: {
        getItem: () => null,
        setItem: () => undefined,
        removeItem: () => undefined,
      },
    },
  });
  t.after(() => {
    Object.assign(globalThis, { window: originalWindow, fetch: originalFetch });
  });
  let access: string | null = null;
  let refresh: string | null = null;
  let passwordChange = false;
  let response: unknown = { access_token: "owner-access", refresh_token: "owner-refresh" };
  let invokeCalls = 0;
  const invokes: unknown[][] = [];
  let refreshCalls = 0;
  let refreshSucceeds = false;
  let refreshOutcome: "ok" | "rejected" | "unavailable" | null = null;
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
    refreshSessionOutcome: async () => {
      if (refreshOutcome) {
        refreshCalls++;
        return refreshOutcome;
      }
      return (await session.refreshSession()) ? "ok" : "rejected";
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
      invoke: async (command: string, args?: unknown) => {
        invokes.push([command, args]);
        if (command === "desktop_mark_signed_out") return void (shell.signedOut = true);
        if (command === "desktop_clear_signed_out") return void (shell.signedOut = false);
        assert.equal(command, "desktop_auth");
        invokeCalls++;
        if (shell.signedOut) return { signed_out: true };
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
    auth, guards, session, events, navigations, loginModes, transitions, shell,
    setResponse: (value: unknown) => { response = value; },
    setStatus: (value: typeof status) => { status = value; },
    setRefreshSucceeds: (value: boolean) => { refreshSucceeds = value; },
    setRefreshOutcome: (value: "ok" | "rejected" | "unavailable" | null) => {
      refreshOutcome = value;
    },
    get access() { return access; },
    get refresh() { return refresh; },
    get invokeCalls() { return invokeCalls; },
    invokes,
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

test("a managed session survives forced startup while its refresh gets no answer", async (t) => {
  const f = fixture(t);
  f.session.storeAuthTokens("alice-access", "alice-refresh");
  f.setResponse(multi);
  f.setRefreshOutcome("unavailable");
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
  assert.equal(f.access, "alice-access");
  assert.equal(f.refresh, "alice-refresh");
  assert.deepEqual(f.navigations, []);
  assert.equal(f.auth.getTauriAuthFailure(), null);
  await f.guards.requireAuth();
});

test("a managed session whose refresh is rejected at startup still opens login", async (t) => {
  const f = fixture(t);
  f.session.storeAuthTokens("alice-access", "alice-refresh");
  f.setResponse(multi);
  f.setRefreshOutcome("rejected");
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
  assert.equal(f.access, null);
  assert.equal(f.refresh, null);
  assert.deepEqual(f.navigations, [{ to: "/login", replace: true }]);
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

const SINGLE_SEEDED = { initialized: true, requires_password_change: true, login_mode: "single" };

test("an explicit desktop logout keeps the shell secret from signing the owner back in", async (t) => {
  const f = fixture(t);
  // A desktop owner who never set a password: the server still reports the seeded one.
  f.setStatus(SINGLE_SEEDED);
  await f.auth.markDesktopSignedOut();
  assert.equal(f.shell.signedOut, true);
  assert.equal(f.auth.isTauriLoginRequired(), true);
  assert.equal(await f.auth.tauriAutoAuth({ force: true }), true);
  assert.equal(await f.auth.tauriAutoAuth(), false);
  assert.equal(f.access, null);
  assert.deepEqual(f.navigations, [
    { to: "/login", replace: true },
    { to: "/login", replace: true },
  ]);
  assert.equal(f.auth.getTauriAuthFailure(), null);
  // The seeded-password setup page is no way back in for the desktop, so the guards keep login.
  await f.guards.requireGuest();
  await assert.rejects(f.guards.requireAuth(), { to: "/login" });
  await assert.rejects(f.guards.requirePasswordChangeFlow(), { to: "/login" });
});

test("a quit right after Log out cannot restore a session the webview never finished removing", async (t) => {
  const first = fixture(t);
  await first.auth.markDesktopSignedOut();
  // WebView2 lost the token removal: the relaunch still holds the old session, but the shell holds the marker.
  const relaunched = fixture(t, true, first.shell);
  relaunched.session.storeAuthTokens("stale-owner", "stale-refresh");
  relaunched.setRefreshSucceeds(true);
  assert.equal(await relaunched.auth.tauriAutoAuth({ force: true }), true);
  assert.equal(relaunched.access, null);
  assert.equal(relaunched.refreshCalls, 0);
  assert.equal(relaunched.auth.isTauriLoginRequired(), true);
  assert.deepEqual(relaunched.navigations, [{ to: "/login", replace: true }]);
});

test("Continue as owner clears the sign-out and signs a single-account owner in", async (t) => {
  const f = fixture(t);
  f.setStatus(SINGLE_SEEDED);
  await f.auth.markDesktopSignedOut();
  await f.auth.tauriAutoAuth({ force: true });
  assert.equal(await f.auth.continueAsDesktopOwner(), true);
  assert.equal(f.shell.signedOut, false);
  assert.equal(f.access, "owner-access");
  assert.equal(f.auth.isTauriLoginRequired(), false);
  assert.deepEqual(f.transitions, [{ account: "unsloth", route: "/chat" }]);
});

test("Continue as owner on a multi-account install leaves the login form up", async (t) => {
  const f = fixture(t);
  f.setResponse(multi);
  await f.auth.markDesktopSignedOut();
  assert.equal(await f.auth.continueAsDesktopOwner(), false);
  assert.equal(f.access, null);
  assert.equal(f.auth.isTauriLoginRequired(), true);
});

function sidebarSignOut(scope: Record<string, unknown>): () => Promise<void> {
  const source = readFileSync(new URL("../src/components/app-sidebar.tsx", import.meta.url), "utf8");
  const tree = ts.createSourceFile("app-sidebar.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let handler = "";
  const visit = (node: ts.Node) => {
    if (ts.isFunctionDeclaration(node) && node.name?.text === "signOut") handler = node.getText(tree);
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(handler);
  const { outputText } = ts.transpileModule(`return (${handler});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  return new Function(...Object.keys(scope), outputText)(...Object.values(scope));
}

type SignOutCase = { tauri: boolean; owner: boolean; status?: unknown; offline?: boolean; markerFails?: boolean };
async function runSignOut({ tauri, owner, status, offline, markerFails }: SignOutCase): Promise<string[]> {
  const events: string[] = [];
  await sidebarSignOut({
    isTauri: tauri,
    isOwner: owner,
    window: { dispatchEvent: () => true },
    Event,
    AUTH_SESSION_ENDING_EVENT: "ending",
    fetchAuthStatus: async () => {
      events.push("status");
      if (status === undefined) throw new Error("offline");
      return status;
    },
    toast: { error: () => events.push("toast") },
    t: (key: string) => key,
    useSettingsDialogStore: { getState: () => ({ openDialog: (tab: string) => events.push(`open:${tab}`) }) },
    markDesktopSignedOut: async () => {
      if (markerFails) throw new Error("permission denied");
      events.push("signed-out");
    },
    logout: async () => {
      events.push("logout");
      if (offline) throw new TypeError("Failed to fetch");
    },
    clearAuthTokens: () => events.push("clear"),
    navigate: ({ to }: { to: string }) => events.push(`navigate:${to}`),
  })();
  return events;
}

test("desktop Log out suppresses auto-auth before revoking, and still clears the session offline", async () => {
  const seeded = { login_mode: "single", requires_password_change: true };
  assert.deepEqual(await runSignOut({ tauri: true, owner: true, status: seeded }), [
    "status", "signed-out", "logout", "navigate:/login",
  ]);
  assert.deepEqual(await runSignOut({ tauri: true, owner: true, status: seeded, offline: true }), [
    "status", "signed-out", "logout", "clear", "navigate:/login",
  ]);
  // A managed account has no seeded password to worry about; the web build has no shell secret to hold back.
  assert.deepEqual(await runSignOut({ tauri: true, owner: false }), ["signed-out", "logout", "navigate:/login"]);
  assert.deepEqual(await runSignOut({ tauri: false, owner: true }), ["logout", "navigate:/login"]);
});

test("a multi-account desktop owner on the seeded password sets one before Log out", async () => {
  const events = await runSignOut({
    tauri: true,
    owner: true,
    status: { login_mode: "multi", requires_password_change: true },
  });
  assert.deepEqual(events, ["status", "toast", "open:remote-lan"]);
});

test("desktop Log out keeps the session when the password state or the marker is unknown", async () => {
  // Status unreachable: a seeded multi-account owner would be locked out.
  assert.deepEqual(await runSignOut({ tauri: true, owner: true }), ["status", "toast"]);
  // Marker not written: a relaunch would sign straight back in.
  const single = { login_mode: "single", requires_password_change: false };
  assert.deepEqual(await runSignOut({ tauri: true, owner: true, status: single, markerFails: true }), [
    "status", "toast",
  ]);
  assert.deepEqual(await runSignOut({ tauri: true, owner: false, markerFails: true }), ["toast"]);
});

test("a second desktop window cannot auto-sign in after another window logs out", async (t) => {
  const first = fixture(t);
  await first.auth.markDesktopSignedOut();
  const second = fixture(t, true, first.shell);
  // API recovery in the other window, after its tokens were cleared under it.
  assert.equal(await second.auth.tauriAutoAuth(), false);
  assert.equal(second.access, null);
});
