// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One model load prepares the same token three times: the progress pollers, validateModel
// and loadModel. Each preparation used to be its own POST /api/hub/token/validate, three
// sequential round trips on the load's critical path. Drive the real module and count.

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// The module registers its logout listener at import time, so the window has to exist
// before loadWithStubs runs and has to keep the registration for the test to fire it.
const { fireWindowEvent } = installLocalStorageFake();

type ValidationStatus = "valid" | "invalid" | "unavailable" | "missing";

type Prepared = { proceed: boolean; token: string | null };

type ConfirmToken = {
  prepareHfTokenForUse: (token: string | null) => Promise<Prepared>;
  forgetHfTokenValidation: (token?: string) => void;
};

const SESSION_CLEARED = "unsloth:auth-session-cleared";

// The store the module subscribes to, so a test can drive a Settings token change.
function tokenStoreStub() {
  let listener: ((state: { token: string }) => void) | null = null;
  return {
    store: {
      getState: () => ({ token: null, clearToken: () => {} }),
      subscribe: (fn: (state: { token: string }) => void) => {
        listener = fn;
        return () => {
          listener = null;
        };
      },
    },
    change(token: string) {
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
          if (gate) {
            await gate.promise;
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

  // The sequential case is the real one: pollers, then validateModel, then loadModel.
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
  // "unavailable" proves nothing, so caching it would suppress the warning dialog for a
  // token that is in fact bad.
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
  // The cache holds the raw credential, so it must not outlive the session that made it.
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
  // Clearing the maps cannot cancel a request already in flight; without a generation
  // the late resolution writes the raw token back and expiry never runs to remove it.
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
