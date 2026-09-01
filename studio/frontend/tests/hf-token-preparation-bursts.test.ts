// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One model load prepares the same token three times: the progress pollers, validateModel
// and loadModel. Each preparation used to be its own POST /api/hub/token/validate, three
// sequential round trips on the load's critical path. Drive the real module and count.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type ValidationStatus = "valid" | "invalid" | "unavailable" | "missing";

type Prepared = { proceed: boolean; token: string | null };

type ConfirmToken = {
  prepareHfTokenForUse: (token: string | null) => Promise<Prepared>;
  forgetHfTokenValidation: (token?: string) => void;
};

function load(status: ValidationStatus, calls: { n: number }): ConfirmToken {
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
      "@/features/hub/stores/hf-token-store": { useHfTokenStore: noopStore },
      "@/features/settings/stores/settings-dialog-store": {
        useSettingsDialogStore: noopStore,
      },
      "./api": {
        validateHfToken: async () => {
          calls.n += 1;
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
