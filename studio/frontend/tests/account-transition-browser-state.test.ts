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
  ACCOUNT_CHANGED_EVENT,
  ACCOUNT_SCOPED_STORAGE_KEYS,
  legacyBrowserDataBelongsToCurrentAccount,
  notifyAccountAuthenticated,
  purgeAccountScopedBrowserState,
} = await import("../src/lib/account-transition.ts");

function seedAlicesBrowser(): void {
  store.clear();
  for (const key of ACCOUNT_SCOPED_STORAGE_KEYS) store.set(key, "alice-value");
  store.set("chat-draft:thread-1", "unsent private text");
  store.set("chat-draft-pastes:thread-1", "[\"pasted\"]");
  // Not account content: a preference the next person is welcome to.
  store.set("unsloth_settings_active_tab", "general");
}

test("the installation owner keeps this browser's pre-accounts data", () => {
  seedAlicesBrowser();
  assert.equal(notifyAccountAuthenticated("alice", "alice"), false);
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), true);
  // Nothing cleared: it is the same person who left it there.
  assert.equal(store.get("unsloth_hf_token"), "alice-value");
});

test("an upgraded browser that nobody has signed in to keeps its migrations", () => {
  seedAlicesBrowser();
  // The ordinary upgrade path: one user, a live session, and no sign-in through
  // the form. The data is theirs and the legacy imports must still run.
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), true);
});

test("a managed account signing in first does not inherit the legacy data", () => {
  seedAlicesBrowser();
  // On an upgraded install a managed account can reach the login page before
  // the owner does. Claiming the data there would let it migrate the original
  // user's conversations, prompts and Hub credential into its own workspace,
  // and leaving the keys in place would let its stores hydrate from them.
  assert.equal(notifyAccountAuthenticated("bob", "alice"), true);
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), false);
  assert.equal(store.get("unsloth_hf_token"), undefined);
  assert.equal(store.get("chat-draft:thread-1"), undefined);

  // Held, not deleted: it is the previous single user's, and the owner gets it
  // back rather than losing it to whoever reached the login page first.
  assert.equal(notifyAccountAuthenticated("alice", "alice"), true);
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), true);
  assert.equal(store.get("unsloth_hf_token"), "alice-value");
  assert.equal(store.get("chat-draft:thread-1"), "unsent private text");
});

test("chat deletion preferences do not carry into the next account", () => {
  seedAlicesBrowser();
  store.set("unsloth_chat_preferences", "{\"alwaysDeleteChatFiles\":true}");
  notifyAccountAuthenticated("alice", "alice");
  notifyAccountAuthenticated("bob", "alice");
  assert.equal(store.get("unsloth_chat_preferences"), undefined);
});

test("hub search terms do not survive an account change", () => {
  seedAlicesBrowser();
  store.set("unsloth.hub.recentSearches", "[\"alice/private-model\"]");
  notifyAccountAuthenticated("alice", "alice");
  notifyAccountAuthenticated("bob", "alice");
  assert.equal(store.get("unsloth.hub.recentSearches"), undefined);
});

test("a different account gets neither the content nor the migration input", () => {
  seedAlicesBrowser();
  notifyAccountAuthenticated("alice", "alice");
  assert.equal(notifyAccountAuthenticated("bob", "alice"), true);

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
  notifyAccountAuthenticated("alice", "alice");
  notifyAccountAuthenticated("bob", "alice");
  assert.equal(notifyAccountAuthenticated("alice", "alice"), true);
  assert.equal(legacyBrowserDataBelongsToCurrentAccount(), true);
});

test("the same account signing in again clears nothing", () => {
  seedAlicesBrowser();
  notifyAccountAuthenticated("alice", "alice");
  assert.equal(notifyAccountAuthenticated("ALICE ", "alice"), false);
  assert.equal(store.get("chat-draft:thread-1"), "unsent private text");
});

test("an account change tells the live stores to reset themselves", () => {
  const seen: string[] = [];
  const listeners = new Map<string, (event: unknown) => void>();
  (globalThis as { window?: Record<string, unknown> }).window = {
    ...((globalThis as { window?: Record<string, unknown> }).window ?? {}),
    dispatchEvent: (event: { type: string }) => {
      seen.push(event.type);
      listeners.get(event.type)?.(event);
      return true;
    },
  } as Record<string, unknown>;
  (globalThis as { Event?: unknown }).Event ??= class {
    type: string;
    constructor(type: string) {
      this.type = type;
    }
  };

  seedAlicesBrowser();
  notifyAccountAuthenticated("alice", "alice");
  notifyAccountAuthenticated("bob", "alice");
  // Clearing a persisted key does not change an already-hydrated store: a
  // same-window removeItem fires no storage event, and the SPA does not reload
  // between accounts.
  assert.deepEqual(seen, [ACCOUNT_CHANGED_EVENT]);
});

test("the purge is inverted, so a store nobody listed is still cleared", () => {
  store.clear();
  store.set("unsloth_hf_token", "hf_alice");
  // None of these four were on any list. Each was reported separately, which is
  // the reason the rule is now "clear unless it is chrome".
  store.set("unsloth.studio.downloads", "[alice's private repo jobs]");
  store.set("unsloth_model_configs", "{alice's templates and args}");
  store.set("unsloth_pinned_chats", "[alice's chats]");
  store.set("unsloth_reasoning_effort", "high");
  // Chrome, which survives so the next account does not get a reset browser.
  store.set("unsloth_locale", "en");
  store.set("unsloth.hub.modelsTab", "installed");
  store.set("unsloth_onboarding_done", "1");
  // Not ours to delete.
  store.set("some-other-app.state", "keep");

  purgeAccountScopedBrowserState();

  assert.equal(store.get("unsloth_hf_token"), undefined);
  assert.equal(store.get("unsloth.studio.downloads"), undefined);
  assert.equal(store.get("unsloth_model_configs"), undefined);
  assert.equal(store.get("unsloth_pinned_chats"), undefined);
  assert.equal(store.get("unsloth_reasoning_effort"), undefined);
  assert.equal(store.get("unsloth_locale"), "en");
  assert.equal(store.get("unsloth.hub.modelsTab"), "installed");
  assert.equal(store.get("unsloth_onboarding_done"), "1");
  assert.equal(store.get("some-other-app.state"), "keep");
});

test("the session and this module's own markers are never purged", () => {
  store.clear();
  store.set("unsloth_auth_token", "the new account's session");
  store.set("unsloth.browser-account.v1", "bob");
  store.set("unsloth.legacy-data-owner.v1", "unsloth");
  store.set("unsloth.legacy-quarantine.v1", "{held}");

  purgeAccountScopedBrowserState();

  // Purging the token would sign the incoming account straight back out, and
  // dropping the markers would re-arm the legacy migration for whoever is next.
  assert.equal(store.get("unsloth_auth_token"), "the new account's session");
  assert.equal(store.get("unsloth.browser-account.v1"), "bob");
  assert.equal(store.get("unsloth.legacy-data-owner.v1"), "unsloth");
  assert.equal(store.get("unsloth.legacy-quarantine.v1"), "{held}");
});
