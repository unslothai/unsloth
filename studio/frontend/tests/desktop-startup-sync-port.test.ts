// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Desktop mounts the root layout before the backend port arrives. The startup pin and model
// override sync must wait for the Desktop shell, or every retry goes to the placeholder port.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type AuthApi = {
  authFetch: (input: string, init?: RequestInit) => Promise<Response>;
};

test("a Desktop request started before the port arrives never reaches the real port", async (t) => {
  let port: number | null = null;
  const authApi = loadWithStubs<AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": {
        apiUrl: (path: string) => `http://127.0.0.1:${port ?? 0}${path}`,
        getApiPort: () => port,
        isTauri: true,
      },
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "./session": {
        clearAuthTokens: () => {},
        getAuthToken: () => null,
        getRefreshToken: () => null,
        mustChangePassword: () => false,
        setMustChangePassword: () => {},
        storeAuthTokens: () => {},
      },
      "@tauri-apps/api/core": { invoke: async () => false },
    },
  );

  const urls: string[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    urls.push(url);
    // The backend reports its port while the first request is in flight.
    port = 61801;
    if (url.startsWith("http://127.0.0.1:0/")) throw new TypeError("fetch failed");
    return new Response("{}", { status: 200 });
  };
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const outcome = authApi
      .authFetch("/api/settings/pinned-models")
      .then(() => "ok", () => "failed");
    let settled: string | null = null;
    void outcome.then((value) => {
      settled = value;
    });
    for (let i = 0; i < 200 && settled === null; i++) {
      await new Promise((resolve) => setImmediate(resolve));
      t.mock.timers.tick(1_000);
    }
    assert.equal(settled, "failed");
    // The URL is resolved once: all six attempts stay on port 0 after the real port is known.
    assert.equal(urls.length, 6);
    assert.ok(urls.every((url) => url.startsWith("http://127.0.0.1:0/")), urls.join(", "));
  } finally {
    t.mock.timers.reset();
    globalThis.fetch = originalFetch;
  }
});

test("the startup pin and override sync waits for the Desktop shell", async () => {
  const root = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  const start = root.indexOf("void backfillModelOverrides();");
  assert.ok(start > 0, "root no longer starts the override backfill");
  const effectStart = root.lastIndexOf("useEffect(() => {", start);
  const effectEnd = root.indexOf("}, [", start);
  const effect = root.slice(effectStart, root.indexOf("]);", effectEnd) + 3);
  assert.match(effect, /void hydratePins\(\);/);
  assert.match(
    effect,
    /if \(isAuthFlowRoute \|\| !desktopShellReady\) \{\s*return;\s*\}/,
    "the sync must not start before the Desktop shell is ready",
  );
  assert.match(effect, /\}, \[isAuthFlowRoute, desktopShellReady\]\);$/);
  // Declared before the effect: its deps are read during render.
  assert.ok(root.indexOf("const desktopShellReady = useDesktopShellReady();") < effectStart);

  // The web build is unchanged: the shell gate is always open there.
  const shell = await readFile(
    new URL("../src/app/desktop-shell-ready.ts", import.meta.url),
    "utf8",
  );
  assert.match(shell, /let ready = !isTauri;/);
  // And on Desktop it only opens once the backend runs (port known) and sign-in finished.
  const provider = await readFile(new URL("../src/app/provider.tsx", import.meta.url), "utf8");
  assert.match(provider, /const canMountApp = status === "running" && desktopAuthReady;/);
  assert.match(provider, /setDesktopShellReady\(canMountApp && appShellReady\);/);
});
