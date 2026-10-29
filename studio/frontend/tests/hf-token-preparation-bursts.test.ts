// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One model load prepares the same token three times; these count the validate round trips.

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// The module registers its logout listener at import time, so the window must exist first.
const { fireWindowEvent } = installLocalStorageFake();

type ValidationStatus = "valid" | "invalid" | "unavailable" | "missing";

type Prepared = { proceed: boolean; token: string | null };

type ConfirmToken = {
  prepareHfTokenForUse: (token: string | null) => Promise<Prepared>;
  forgetHfTokenValidation: (token?: string) => void;
};

const SESSION_CLEARED = "unsloth:auth-session-cleared";

function tokenStoreStub(initial: string | null = null) {
  let listener: ((state: { token: string }) => void) | null = null;
  let current = initial;
  return {
    store: {
      getState: () => ({ token: current, clearToken: () => {} }),
      subscribe: (fn: (state: { token: string }) => void) => {
        listener = fn;
        return () => {
          listener = null;
        };
      },
    },
    change(token: string) {
      current = token;
      listener?.({ token });
    },
  };
}

type Gate = { promise: Promise<void>; open: () => void };

function makeGate(): Gate {
  let open: () => void = () => {};
  const promise = new Promise<void>((resolve) => {
    open = () => resolve();
  });
  return { promise, open };
}

function load(
  status: ValidationStatus,
  calls: { n: number },
  tokenStore = tokenStoreStub(),
  gate: Gate | null = null,
  gates: Gate[] | null = null,
): ConfirmToken {
  const noopStore = {
    getState: () => ({
      token: null,
      clearToken: () => {},
      openDialog: () => {},
      requestDecision: async () => "cancel",
    }),
  };
  return loadWithStubs<ConfirmToken>(
    new URL("../src/features/hf-auth/confirm-token.ts", import.meta.url),
    {
      "@/features/auth/session-events": {
        AUTH_SESSION_CLEARED_EVENT: SESSION_CLEARED,
      },
      "@/features/hub/stores/hf-token-store": {
        useHfTokenStore: tokenStore.store,
      },
      "@/features/settings/stores/settings-dialog-store": {
        useSettingsDialogStore: noopStore,
      },
      "./api": {
        validateHfToken: async () => {
          calls.n += 1;
          const own = gates ? gates[calls.n - 1] : gate;
          if (own) {
            await own.promise;
          }
          return { status, retryAfterSeconds: null };
        },
      },
      "./store": { useHfTokenWarningStore: noopStore },
    },
  );
}

test("a burst of preparations for one valid token validates once", async () => {
  const calls = { n: 0 };
  const mod = load("valid", calls);

  const results = await Promise.all([
    mod.prepareHfTokenForUse("hf_valid"),
    mod.prepareHfTokenForUse("hf_valid"),
    mod.prepareHfTokenForUse("hf_valid"),
  ]);

  assert.equal(calls.n, 1, "one load issued more than one validation round trip");
  for (const result of results) {
    assert.deepEqual(result, { proceed: true, token: "hf_valid" });
  }

  await mod.prepareHfTokenForUse("hf_valid");
  assert.equal(calls.n, 1);
});

test("distinct tokens are validated separately", async () => {
  const calls = { n: 0 };
  const mod = load("valid", calls);

  await mod.prepareHfTokenForUse("hf_one");
  await mod.prepareHfTokenForUse("hf_two");

  assert.equal(calls.n, 2, "two different credentials shared one verdict");
});

test("a non-definitive verdict is never reused", async () => {
  // "unavailable" proves nothing, so caching it would hide the warning for a bad token.
  const calls = { n: 0 };
  const mod = load("unavailable", calls);

  await mod.prepareHfTokenForUse("hf_maybe");
  await mod.prepareHfTokenForUse("hf_maybe");

  assert.equal(calls.n, 2, "an inconclusive verdict was cached");
});

test("an invalid verdict is re-checked rather than remembered", async () => {
  const calls = { n: 0 };
  const mod = load("invalid", calls);

  const first = await mod.prepareHfTokenForUse("hf_bad");
  const second = await mod.prepareHfTokenForUse("hf_bad");

  assert.equal(first.proceed, false);
  assert.equal(second.proceed, false);
  assert.equal(calls.n, 2, "an invalid verdict was cached and the dialog skipped");
});

test("forgetting a token drops an unexpired window", async () => {
  const calls = { n: 0 };
  const mod = load("valid", calls);

  await mod.prepareHfTokenForUse("hf_valid");
  mod.forgetHfTokenValidation("hf_valid");
  await mod.prepareHfTokenForUse("hf_valid");

  assert.equal(calls.n, 2, "a replaced credential rode the previous window");
});


test("a logout drops the cached bearer token", async () => {
  const calls = { n: 0 };
  const mod = load("valid", calls);

  await mod.prepareHfTokenForUse("hf_valid");
  assert.equal(calls.n, 1);

  const delivered = fireWindowEvent(SESSION_CLEARED, {});
  assert.ok(delivered > 0, "the module registered no logout listener");

  await mod.prepareHfTokenForUse("hf_valid");
  assert.equal(calls.n, 2, "a previous session's credential survived the logout");
});

test("replacing the stored credential drops the superseded key", async () => {
  const calls = { n: 0 };
  const tokenStore = tokenStoreStub();
  const mod = load("valid", calls, tokenStore);

  // The subscription is installed on first use, so prepare before staging the change.
  await mod.prepareHfTokenForUse("hf_old");
  assert.equal(calls.n, 1);
  tokenStore.change("hf_old");

  tokenStore.change("hf_new");

  await mod.prepareHfTokenForUse("hf_old");
  assert.equal(calls.n, 2, "the replaced credential kept its window");
});


test("a logout mid-validation does not let the reply repopulate the cache", async () => {
  // Clearing the maps cannot cancel an in-flight request; a generation stops its late write.
  const calls = { n: 0 };
  const gate = makeGate();
  const mod = load("valid", calls, tokenStoreStub(), gate);

  const pending = mod.prepareHfTokenForUse("hf_valid");
  mod.forgetHfTokenValidation();
  gate.open();
  await pending;
  assert.equal(calls.n, 1);

  await mod.prepareHfTokenForUse("hf_valid");
  assert.equal(calls.n, 2, "the in-flight reply repopulated the cleared cache");
});


test("replacing the token that was already stored drops its window", async () => {
  // zustand does not fire subscribe on install, so lastKnownStoredToken must be seeded.
  const calls = { n: 0 };
  const tokenStore = tokenStoreStub("hf_old");
  const mod = load("valid", calls, tokenStore);

  await mod.prepareHfTokenForUse("hf_old");
  assert.equal(calls.n, 1);

  tokenStore.change("hf_new");

  await mod.prepareHfTokenForUse("hf_old");
  assert.equal(calls.n, 2, "the superseded credential kept its window");
});


test("a replacement request is not evicted by the old one settling", async () => {
  // "unavailable" is never cached, so sharing is observable purely through the count.
  const calls = { n: 0 };
  const gates = [makeGate(), makeGate(), makeGate()];
  const mod = load("unavailable", calls, tokenStoreStub(), null, gates);

  const stale = mod.prepareHfTokenForUse("hf_a");
  mod.forgetHfTokenValidation("hf_a");

  const replacement = mod.prepareHfTokenForUse("hf_a");
  assert.equal(calls.n, 2, "the replacement did not start its own request");

  gates[0].open();
  await stale;

  const shared = mod.prepareHfTokenForUse("hf_a");
  assert.equal(calls.n, 2, "the settling request evicted its live replacement");

  gates[1].open();
  await Promise.all([replacement, shared]);
});
