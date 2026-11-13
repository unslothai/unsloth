// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type * as AuthApi from "../src/features/auth/api.ts";

for (const mode of ["single", "multi"]) {
  test(`${mode} auth redirect reconciles the correct password-change requirement`, async (t) => {
    const previousWindow = globalThis.window;
    const previousFetch = globalThis.fetch;
    let change = false;
    const location = { pathname: "/chat", href: "" };
    globalThis.window = { location } as unknown as Window & typeof globalThis;
    globalThis.fetch = async (path) =>
      String(path).endsWith("/status")
        ? Response.json({ login_mode: mode, requires_password_change: false })
        : Response.json(
            { detail: "Password change required" },
            { status: 403 },
          );
    t.after(() => {
      globalThis.window = previousWindow;
      globalThis.fetch = previousFetch;
    });
    const api = loadWithStubs<typeof AuthApi>(
      new URL("../src/features/auth/api.ts", import.meta.url),
      {
        "@/lib/api-base": { apiUrl: (path: string) => path, isTauri: false },
        "@/lib/account-transition": { accountTransitionPending: () => false },
        "./session": {
          getAuthToken: () => "setup-token",
          getRefreshToken: () => null,
          mustChangePassword: () => change,
          setMustChangePassword: (required: boolean) => {
            change = required;
          },
        },
      },
    );
    assert.equal((await api.authFetch("/api/accounts")).status, 403);
    await new Promise<void>((resolve) => setImmediate(resolve));
    assert.equal(
      location.href,
      mode === "multi" ? "/change-password" : "/login",
    );
    assert.equal(change, mode === "multi");
  });
}

test("a query chat's text goes through a stale-token sign-in", async (t) => {
  const previousWindow = globalThis.window;
  const previousFetch = globalThis.fetch;
  const location = { pathname: "/chat", search: "?new=query-1", href: "" };
  const drafts = await import("../src/features/chat/utils/composer-draft.ts");
  const slot = drafts.composerDraftKey("query-1");
  assert.ok(drafts.isQueryDraftId("query-1"));
  globalThis.window = {
    location,
    sessionStorage: { getItem: (key: string) => (key === slot ? 'what is "lora"?' : null) },
  } as unknown as Window & typeof globalThis;
  globalThis.fetch = async (path) =>
    String(path).endsWith("/status")
      ? Response.json({ login_mode: "single", requires_password_change: false })
      : Response.json({ detail: "expired" }, { status: 401 });
  t.after(() => {
    globalThis.window = previousWindow;
    globalThis.fetch = previousFetch;
  });
  const api = loadWithStubs<typeof AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": { apiUrl: (path: string) => path, isTauri: false },
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "./session": {
        getAuthToken: () => "stale-token",
        getRefreshToken: () => null,
        mustChangePassword: () => false,
        setMustChangePassword: () => {},
        clearAuthTokens: () => {},
      },
    },
  );
  await api.authFetch("/api/chat/sidebar/rows");
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(location.href, `/login?q=${encodeURIComponent('what is "lora"?')}`);
});
