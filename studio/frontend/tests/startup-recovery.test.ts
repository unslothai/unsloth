// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { loadWithStubs } from "./helpers/module-stubs.ts";
import { installLocalStorageFake, registerStoreStubResolver } from "./helpers/kit.ts";

type Bootstrap = typeof import("../src/features/credentials/bootstrap.ts");
const source = process.env.CREDENTIAL_BOOTSTRAP_ROOT
  ? pathToFileURL(`${process.env.CREDENTIAL_BOOTSTRAP_ROOT}/src/features/credentials/`)
  : new URL("../src/features/credentials/", import.meta.url);

function fixture() {
  const session = { epoch: 0, token: true };
  const live = { hf: true, providers: true };
  const calls = { hf: 0, providers: 0 };
  const state = { hfHydrated: false, providers: [] as string[] };
  let holdHf: Promise<void> | null = null;
  let holdProviders: Promise<string[]> | null = null;
  const reconciliation = loadWithStubs(
    new URL("reconciliation.ts", source), {},
  );
  const mod = loadWithStubs<Bootstrap>(new URL("bootstrap.ts", source), {
    "@/features/auth": {
      getAuthSessionEpoch: () => session.epoch,
      hasAuthToken: () => session.token,
    },
    "@/features/chat/stores/external-providers-store": {
      useExternalProvidersStore: {
        getState: () => ({
          providers: state.providers,
          setProviders: (rows: string[]) => { state.providers = rows; },
        }),
      },
    },
    "@/features/chat/sync-external-providers": {
      syncExternalProvidersFromBackend: async () => {
        calls.providers += 1;
        if (!live.providers) throw new Error("providers temporarily unavailable");
        if (holdProviders) return holdProviders;
        return ["saved-provider"];
      },
    },
    "@/features/hub/stores/hf-token-store": {
      hydrateHfTokenFromBackend: async () => {
        calls.hf += 1;
        if (holdHf) await holdHf;
        if (!live.hf) throw new Error("HF temporarily unavailable");
        state.hfHydrated = true;
      },
    },
    "./reconciliation": reconciliation,
  });
  return {
    mod, session, live, calls, state,
    holdHf: (value: Promise<void>) => { holdHf = value; },
    holdProviders: (value: Promise<string[]>) => { holdProviders = value; },
  };
}

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

test("successful startup work is reused before and after settlement", async () => {
  for (const settled of [false, true]) {
    const f = fixture();
    f.mod.startStartupCredentialBootstrap();
    if (settled) await settle();
    await f.mod.takeStartupCredentialBootstrap();
    assert.deepEqual(f.calls, { hf: 1, providers: 1 });
    assert.equal(f.state.hfHydrated, true);
    assert.deepEqual(f.state.providers, ["saved-provider"]);
    assert.equal(f.mod.takeStartupCredentialBootstrap(), null);
  }
});

for (const failure of ["hf", "providers", "both"] as const) {
  test(`the gate retries a settled ${failure} startup failure after recovery`, async () => {
    const f = fixture();
    f.live.hf = failure === "providers";
    f.live.providers = failure === "hf";
    f.mod.startStartupCredentialBootstrap();
    await settle();
    f.live.hf = f.live.providers = true;
    await f.mod.takeStartupCredentialBootstrap();
    assert.equal(f.state.hfHydrated, true);
    assert.deepEqual(f.state.providers, ["saved-provider"]);
    assert.deepEqual(f.calls, {
      hf: failure === "providers" ? 1 : 2,
      providers: failure === "hf" ? 1 : 2,
    });
    assert.equal(f.mod.takeStartupCredentialBootstrap(), null);
  });
}

test("a failure that settles after handoff also gets one retry", async () => {
  const f = fixture();
  let fail!: (reason: Error) => void;
  f.holdHf(new Promise<void>((_, reject) => { fail = reject; }));
  f.mod.startStartupCredentialBootstrap();
  const gate = f.mod.takeStartupCredentialBootstrap();
  f.holdHf(Promise.resolve());
  fail(new Error("initial request failed"));
  await gate;
  assert.equal(f.state.hfHydrated, true);
  assert.deepEqual(f.calls, { hf: 2, providers: 1 });
  assert.deepEqual(f.state.providers, ["saved-provider"]);
});

test("a session change while startup is pending prevents its retry", async () => {
  const f = fixture();
  let fail!: (reason: Error) => void;
  f.holdHf(new Promise<void>((_, reject) => { fail = reject; }));
  f.live.providers = false;
  f.mod.startStartupCredentialBootstrap();
  const gate = f.mod.takeStartupCredentialBootstrap();
  f.session.epoch += 1;
  f.holdHf(Promise.resolve());
  f.live.providers = true;
  fail(new Error("old session request failed"));
  await gate;
  assert.deepEqual(f.calls, { hf: 1, providers: 1 });
  assert.deepEqual(f.state.providers, []);
});

test("retried providers are not published before the HF retry settles", async () => {
  const f = fixture();
  f.live.hf = f.live.providers = false;
  f.mod.startStartupCredentialBootstrap();
  await settle();
  f.live.hf = f.live.providers = true;
  let finish!: () => void;
  f.holdHf(new Promise<void>((resolve) => { finish = resolve; }));
  const gate = f.mod.takeStartupCredentialBootstrap();
  await settle();
  assert.deepEqual(f.calls, { hf: 2, providers: 2 });
  assert.deepEqual(f.state.providers, []);
  finish();
  await gate;
  assert.equal(f.state.hfHydrated, true);
  assert.deepEqual(f.state.providers, ["saved-provider"]);
});

test("a failed handoff retry settles without a retry loop", async () => {
  const f = fixture();
  f.live.hf = f.live.providers = false;
  f.mod.startStartupCredentialBootstrap();
  await settle();
  await f.mod.takeStartupCredentialBootstrap();
  assert.deepEqual(f.calls, { hf: 2, providers: 2 });
  assert.equal(f.state.hfHydrated, false);
  assert.deepEqual(f.state.providers, []);
  await f.mod.bootstrapPersistedCredentials();
  assert.deepEqual(f.calls, { hf: 3, providers: 3 });
});

test("a replacement bootstrap suppresses a detached startup retry", async () => {
  const f = fixture();
  f.live.providers = false;
  f.mod.startStartupCredentialBootstrap();
  await settle();
  f.live.providers = true;
  f.holdProviders(Promise.resolve(["replacement"]));
  const detached = f.mod.takeStartupCredentialBootstrap();
  const replacement = f.mod.bootstrapPersistedCredentials();
  let finish!: (providers: string[]) => void;
  f.holdProviders(new Promise((resolve) => { finish = resolve; }));
  await replacement;
  f.state.providers = ["user-edit"];
  finish(["stale-startup-retry"]);
  await detached;
  assert.deepEqual(f.state.providers, ["user-edit"]);
  assert.equal(f.calls.providers, 2);
});

test("a replacement bootstrap supersedes a pending startup publication", async () => {
  const f = fixture();
  let finish!: (providers: string[]) => void;
  f.holdProviders(new Promise((resolve) => { finish = resolve; }));
  f.mod.startStartupCredentialBootstrap();
  const detached = f.mod.takeStartupCredentialBootstrap();
  f.holdProviders(Promise.resolve(["replacement"]));
  await f.mod.bootstrapPersistedCredentials();
  f.state.providers = ["user-edit"];
  finish(["stale-original-startup"]);
  await detached;
  assert.deepEqual(f.state.providers, ["user-edit"]);
  assert.equal(f.calls.providers, 2);
});

test("gate replay recovers a pending shared HF read that fails after readiness", async () => {
  registerStoreStubResolver();
  const { store } = installLocalStorageFake();
  store.set("unsloth_auth_token", "session-token");
  const originalFetch = globalThis.fetch;
  let release!: (response: Response) => void;
  let started!: () => void;
  const firstResponse = new Promise<Response>((resolve) => { release = resolve; });
  const firstStarted = new Promise<void>((resolve) => { started = resolve; });
  let reads = 0;
  globalThis.fetch = (async () => {
    reads += 1;
    if (reads === 1) {
      started();
      return firstResponse;
    }
    return new Response(JSON.stringify({ has_token: true, token: "hf_saved_fixture" }), {
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  try {
    const hf = await import("../src/features/hub/stores/hf-token-store.ts");
    const reconciliation = loadWithStubs(new URL("reconciliation.ts", source), {});
    const mod = loadWithStubs<Bootstrap>(new URL("bootstrap.ts", source), {
      "@/features/auth": { getAuthSessionEpoch: () => 0, hasAuthToken: () => true },
      "@/features/chat/stores/external-providers-store": {
        useExternalProvidersStore: { getState: () => ({ providers: [], setProviders: () => {} }) },
      },
      "@/features/chat/sync-external-providers": {
        syncExternalProvidersFromBackend: async () => [],
      },
      "@/features/hub/stores/hf-token-store": hf,
      "./reconciliation": reconciliation,
    });
    mod.startStartupCredentialBootstrap();
    await firstStarted;
    const firstGate = mod.takeStartupCredentialBootstrap();
    const replayGate = mod.takeStartupCredentialBootstrap() ?? mod.bootstrapPersistedCredentials();
    release(new Response(JSON.stringify({ detail: "brief early outage" }), {
      status: 503, headers: { "Content-Type": "application/json" },
    }));
    await Promise.all([firstGate, replayGate]);
    assert.equal(hf.getHfToken(), "hf_saved_fixture");
    assert.equal(reads, 2);
    assert.equal(mod.takeStartupCredentialBootstrap(), null);
  } finally {
    globalThis.fetch = originalFetch;
    Reflect.deleteProperty(globalThis, "window");
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});
