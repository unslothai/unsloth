// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Bootstrap = typeof import("../src/features/credentials/bootstrap.ts");

function loadBootstrap() {
  const session = { epoch: 0, token: "original" as string | null };
  const runs: Array<{ resolve: () => void; promise: Promise<void> }> = [];
  const mod = loadWithStubs<Bootstrap>(
    new URL("../src/features/credentials/bootstrap.ts", import.meta.url),
    {
      "@/features/auth": {
        getAuthSessionEpoch: () => session.epoch,
        hasAuthToken: () => Boolean(session.token),
      },
      "@/features/chat/stores/external-providers-store": {
        useExternalProvidersStore: {
          getState: () => ({ providers: [], setProviders: () => {} }),
        },
      },
      "@/features/chat/sync-external-providers": {
        syncExternalProvidersFromBackend: async () => [],
      },
      "@/features/hub/stores/hf-token-store": {
        hydrateHfTokenFromBackend: async () => {},
      },
      "./reconciliation": {
        runCredentialBootstrap: () => {
          let resolve = () => {};
          const promise = new Promise<void>((done) => {
            resolve = done;
          });
          runs.push({ resolve, promise });
          return promise;
        },
      },
    },
  );
  return { mod, session, runs };
}

test("the gate's first run joins the run the entry started", async () => {
  const { mod, runs } = loadBootstrap();
  mod.startStartupCredentialBootstrap();
  mod.startStartupCredentialBootstrap();
  assert.equal(runs.length, 1, "a second entry call started a second run");
  const gate = mod.takeStartupCredentialBootstrap();
  runs[0].resolve();
  await gate;
  assert.equal(runs.length, 1);
});

test("gate replay shares the startup handoff until it settles", async () => {
  const { mod, runs } = loadBootstrap();
  mod.startStartupCredentialBootstrap();
  const first = mod.takeStartupCredentialBootstrap();
  assert.equal(mod.takeStartupCredentialBootstrap(), first);
  runs[0].resolve();
  await first;
  assert.equal(mod.takeStartupCredentialBootstrap(), null);
  mod.startStartupCredentialBootstrap();
  assert.equal(runs.length, 1, "a second startup run after the first was taken");
});

test("a settled startup run is still the gate's first run", async () => {
  const { mod, runs } = loadBootstrap();
  mod.startStartupCredentialBootstrap();
  runs[0].resolve();
  await runs[0].promise;
  await mod.takeStartupCredentialBootstrap();
  assert.equal(runs.length, 1, "the gate repeated the startup requests");
});

test("no startup run without a token, so the gate runs as before", () => {
  const { mod, session, runs } = loadBootstrap();
  session.token = null;
  mod.startStartupCredentialBootstrap();
  assert.equal(runs.length, 0);
  session.token = "new-session";
  assert.equal(mod.takeStartupCredentialBootstrap(), null);
  assert.equal(runs.length, 0);
});

test("a session change before the gate mounts drops the startup run", () => {
  for (const change of ["sign-in", "sign-out"] as const) {
    const { mod, session, runs } = loadBootstrap();
    mod.startStartupCredentialBootstrap();
    if (change === "sign-in") session.epoch += 1;
    else {
      session.epoch += 1;
      session.token = null;
    }
    assert.equal(mod.takeStartupCredentialBootstrap(), null, change);
    assert.equal(runs.length, 1, change);
  }
});

test("a refreshed access token keeps the startup run", async () => {
  const { mod, runs, session } = loadBootstrap();
  mod.startStartupCredentialBootstrap();
  session.token = "refreshed";
  const gate = mod.takeStartupCredentialBootstrap();
  assert.notEqual(gate, null);
  runs[0].resolve();
  await gate;
  assert.equal(runs.length, 1);
});

test("the entry starts the credential bootstrap only where the gate will run it", () => {
  const main = readSrc("main.tsx");
  const tail = main.slice(main.indexOf("fetchDeviceType().catch(() => undefined);"));
  assert.match(
    tail,
    /if \(\s*!isTauri &&\s*hasSettledAuthSession\(\) &&\s*startPath !== "\/login" &&\s*startPath !== "\/change-password"\s*\) \{\s*startStartupCredentialBootstrap\(\);\s*\}/,
  );
  const root = readSrc("app/routes/__root.tsx");
  const gate = root.slice(
    root.indexOf("function CredentialBootstrapGate("),
    root.indexOf("const CHAT_ONLY_ALLOWED"),
  );
  assert.match(
    gate,
    /void \(\s*takeStartupCredentialBootstrap\(\) \?\? bootstrapPersistedCredentials\(\)\s*\)\.finally\(/,
  );
  assert.equal(gate.match(/takeStartupCredentialBootstrap\(\)/g)?.length, 1);
  for (const page of ["login", "change-password"]) {
    assert.match(readSrc(`app/routes/${page}.tsx`), /isAuthFlow: true/);
  }
});
