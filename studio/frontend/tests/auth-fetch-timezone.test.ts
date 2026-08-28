// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type AuthApi = {
  authFetch: (input: string, init?: RequestInit) => Promise<Response>;
};

test("authFetch sends the browser timezone", async () => {
  const originalFetch = globalThis.fetch;
  let receivedHeaders = new Headers();
  globalThis.fetch = async (_input, init) => {
    receivedHeaders = new Headers(init?.headers);
    return new Response(null, { status: 200 });
  };

  try {
    const authApi = loadWithStubs<AuthApi>(
      new URL("../src/features/auth/api.ts", import.meta.url),
      {
        "@/lib/api-base": { apiUrl: (path: string) => path, isTauri: false },
        "./session": {
          clearAuthTokens: () => {},
          getAuthToken: () => "access-token",
          getRefreshToken: () => null,
          mustChangePassword: () => false,
          setMustChangePassword: () => {},
          storeAuthTokens: () => {},
        },
      },
    );

    await authApi.authFetch("/api/inference/chat/completions");

    assert.equal(
      receivedHeaders.get("X-Unsloth-Timezone"),
      Intl.DateTimeFormat().resolvedOptions().timeZone,
    );
    assert.equal(
      receivedHeaders.get("X-Unsloth-Timezone-Offset-Minutes"),
      String(new Date().getTimezoneOffset()),
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
