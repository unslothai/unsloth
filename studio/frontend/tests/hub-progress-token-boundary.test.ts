// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Assert on the request the browser emits, not the shape of the source: these still fail
// if the header is dropped, blanked, or eaten by authFetch's own merge.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { hubTokenHeader } from "../src/features/hub/lib/hub-token-header.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type AuthApi = {
  authFetch: (input: string, init?: RequestInit) => Promise<Response>;
};

const read = (relativePath: string): string =>
  readFileSync(new URL(relativePath, import.meta.url), "utf8");

async function emittedHeaders(init?: RequestInit): Promise<Headers> {
  const originalFetch = globalThis.fetch;
  let received = new Headers();
  globalThis.fetch = async (_input, fetchInit) => {
    received = new Headers(fetchInit?.headers);
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
    await authApi.authFetch("/api/models/download-progress?repo_id=org/repo", init);
    return received;
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test("a progress request with a token carries it, alongside the session credential", async () => {
  const headers = await emittedHeaders({ headers: hubTokenHeader("hf_secret") });

  assert.equal(headers.get("X-Unsloth-HF-Token"), "hf_secret");
  // authFetch seeds Headers from init: a caller passing `headers` must not displace these.
  assert.equal(headers.get("Authorization"), "Bearer access-token");
  assert.ok(headers.get("X-Unsloth-Timezone"));
});

test("a tokenless progress request omits the header rather than blanking it", async () => {
  for (const token of [null, undefined, ""]) {
    const headers = await emittedHeaders({ headers: hubTokenHeader(token) });

    // An empty-string header is not equivalent: the backend reads it as present.
    assert.equal(
      headers.has("X-Unsloth-HF-Token"),
      false,
      `token ${JSON.stringify(token)} must not emit the header at all`,
    );
    assert.equal(headers.get("Authorization"), "Bearer access-token");
  }
});

test("hubTokenHeader never leaks the token anywhere but its own header", async () => {
  const headers = await emittedHeaders({ headers: hubTokenHeader("hf_secret") });

  for (const [name, value] of headers.entries()) {
    if (name.toLowerCase() === "x-unsloth-hf-token") continue;
    assert.equal(
      value.includes("hf_secret"),
      false,
      `header ${name} must not carry the Hub token`,
    );
  }
});

test("the progress callers accept and forward a request-scoped token", () => {
  // Whitespace-insensitive: the transport tests cannot prove these callers send one.
  const api = read("../src/features/chat/api/chat-api.ts");
  for (const name of [
    "getGgufDownloadProgress",
    "getDownloadProgress",
    "getDatasetDownloadProgress",
  ]) {
    const start = api.indexOf(`export async function ${name}`);
    assert.notEqual(start, -1, `${name} is missing`);
    const next = api.indexOf("\nexport ", start + 1);
    const body = api.slice(start, next === -1 ? undefined : next);
    assert.match(body, /hfToken\?:\s*string\s*\|\s*null/, `${name} takes no token`);
    assert.match(
      body,
      /headers:\s*hubTokenHeader\(\s*hfToken\s*,?\s*\)/,
      `${name} does not send the token`,
    );
  }
});

test("a local load is not gated behind Hub token preparation", () => {
  // prepareHfTokenForUse validates over the network and can block on a dialog.
  const chatRuntime = read(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  const start = chatRuntime.indexOf("const mayReachHub");
  assert.notEqual(start, -1, "the local-load guard is missing");
  const guarded = chatRuntime.slice(start, start + 600);
  assert.match(guarded, /!isLocal/);
  // An Ollama row is local too, but its id is an opaque reference rather than a path,
  // so isLocalModelPath alone lets it through. chat-load-hub-token-reach.test.ts pins
  // what the predicate itself classifies.
  assert.match(guarded, /!isOllamaLinkPath\(modelId\)/);
  assert.match(guarded, /nativePathToken\s*==\s*null/);
  assert.match(guarded, /if\s*\(mayReachHub\)\s*\{[\s\S]*prepareHfTokenForUse\(hfToken\)/);
});
