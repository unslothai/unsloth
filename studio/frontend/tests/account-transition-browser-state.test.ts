// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  ACCOUNT_SCOPED_STORAGE_KEYS,
  legacyBrowserDataBelongsToCurrentAccount,
  notifyAccountAuthenticated,
} = await import("../src/lib/account-transition.ts");

function seedAlicesBrowser(): void {
  store.clear();
  for (const key of ACCOUNT_SCOPED_STORAGE_KEYS) store.set(key, "alice-value");
  store.set("chat-draft:thread-1", "unsent private text");
  store.set("chat-draft-pastes:thread-1", "[\"pasted\"]");
  // Not account content: a preference the next person is welcome to.
  store.set("unsloth_settings_active_tab", "general");
}

test("the first account to sign in keeps this browser's pre-accounts data", () => {
  seedAlicesBrowser();
  assert.equal(notifyAccountAuthenticated("alice"), false);
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), true);
  // Nothing cleared: it is the same person who left it there.
  assert.equal(store.get("unsloth_hf_token"), "alice-value");
});

test("a different account gets neither the content nor the migration input", () => {
  seedAlicesBrowser();
  notifyAccountAuthenticated("alice");
  assert.equal(notifyAccountAuthenticated("bob"), true);

  for (const key of ACCOUNT_SCOPED_STORAGE_KEYS) {
    assert.equal(store.get(key), undefined, key);
  }
  assert.equal(store.get("chat-draft:thread-1"), undefined);
  assert.equal(store.get("chat-draft-pastes:thread-1"), undefined);
  assert.equal(store.get("unsloth_settings_active_tab"), "general");
  // The legacy chat store, chat settings and Hugging Face token are left on
  // disk for the account they belong to, and simply not offered to this one.
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), false);
});

test("signing back in restores the owner's claim on the legacy data", () => {
  seedAlicesBrowser();
  notifyAccountAuthenticated("alice");
  notifyAccountAuthenticated("bob");
  assert.equal(notifyAccountAuthenticated("alice"), true);
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), true);
});

test("the same account signing in again clears nothing", () => {
  seedAlicesBrowser();
  notifyAccountAuthenticated("alice");
  assert.equal(notifyAccountAuthenticated("ALICE "), false);
  assert.equal(store.get("chat-draft:thread-1"), "unsent private text");
});
