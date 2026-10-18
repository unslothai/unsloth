// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type AuthApi = { authFetch: (...args: unknown[]) => Promise<Response> };
type ImagesApi = {
  generateDiffusionImage: (body: unknown) => Promise<unknown>;
  GenerateResponseLostError: new (message: string) => Error;
};

function loadApis(): ImagesApi {
  const auth = loadWithStubs<AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": {
        apiUrl: (path: string) => path,
        getApiPort: () => 1,
        isTauri: true,
      },
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "./session": {
        clearAuthTokens: () => {},
        getAuthToken: () => "access-token",
        getRefreshToken: () => null,
        mustChangePassword: () => false,
        setMustChangePassword: () => {},
        storeAuthTokens: () => {},
      },
      "@tauri-apps/api/core": { invoke: async () => false },
    },
  );
  return loadWithStubs<ImagesApi>(
    new URL("../src/features/images/api.ts", import.meta.url),
    {
      "@/lib/model-lifecycle-events": { withBackgroundLoadNotice: (f: unknown) => f },
      "@/features/auth": { authFetch: auth.authFetch },
      "@/lib/format-fastapi-error": { readFastApiError: async () => "" },
    },
    { relativePassthrough: true },
  );
}

test("a dropped generate reply is surfaced as lost, not re-sent", async () => {
  const api = loadApis();
  const originalFetch = globalThis.fetch;
  let posts = 0;
  globalThis.fetch = (async () => {
    posts += 1;
    throw new TypeError("Load failed");
  }) as typeof fetch;
  try {
    await assert.rejects(
      api.generateDiffusionImage({ prompt: "x" }),
      (err: Error) => err instanceof api.GenerateResponseLostError,
    );
    assert.equal(posts, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
