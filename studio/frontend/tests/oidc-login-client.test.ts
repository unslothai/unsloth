// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// biome-ignore lint/correctness/noNodejsModules: Node's built-in test runner is the repository test harness.
import assert from "node:assert/strict";
// biome-ignore lint/correctness/noNodejsModules: Node's built-in test runner is the repository test harness.
import test from "node:test";
import type * as LoginClient from "../src/features/auth/login-client.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function loadClient() {
  return loadWithStubs<typeof LoginClient>(
    new URL("../src/features/auth/login-client.ts", import.meta.url),
    {
      "@/lib/api-base": { apiUrl: (path: string) => path },
      "@/lib/account-transition": {
        normalizeAccountUsername: (value: string) => value,
        resetFullAccessForMultiUser: () => undefined,
      },
    },
  );
}

test("public OIDC config exposes only display-safe provider state", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    // biome-ignore lint/style/useNamingConvention: API schema
    Response.json({ enabled: true, display_name: "Company SSO" });
  try {
    assert.deepEqual(await loadClient().fetchOIDCConfig(), {
      enabled: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      display_name: "Company SSO",
    });
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test("an unavailable OIDC config keeps local login UI behavior", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(null, { status: 404 });
  try {
    assert.deepEqual(await loadClient().fetchOIDCConfig(), { enabled: false });
  } finally {
    globalThis.fetch = previousFetch;
  }
});
