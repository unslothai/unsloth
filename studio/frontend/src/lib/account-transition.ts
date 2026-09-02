// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Browser-local state that belongs to one account, and what to do when another
 * one signs in on the same browser.
 *
 * Everything under localStorage and IndexedDB is origin-wide: it survives a
 * logout, and until accounts existed that was correct, because there was only
 * ever one user. With managed accounts the same browser is shared, so a value
 * left behind is either read by the next account (drafts, dictation history,
 * provider metadata, the last-model shadow) or, worse, treated as migration
 * input and copied into that account's own workspace (the legacy chat, chat
 * settings and Hugging Face token imports).
 *
 * Two rules, applied at the one point where a username is admitted:
 *  - content keys are cleared when the account changes;
 *  - legacy migration input is bound to the account that owns this browser's
 *    pre-accounts data, which is whoever first signed in here.
 */

const LAST_ACCOUNT_KEY = "unsloth.browser-account.v1";
const LEGACY_DATA_OWNER_KEY = "unsloth.legacy-data-owner.v1";

/**
 * Dispatched on window when a different account signs in on this browser.
 *
 * Clearing a persisted key does not change an already-hydrated store: a
 * same-window removeItem fires no storage event, and the SPA does not reload
 * between accounts. Stores holding account content listen for this and reset
 * themselves, the way they already reset on a cleared session.
 */
export const ACCOUNT_CHANGED_EVENT = "unsloth:account-changed";

/** Whole keys holding one account's own content or credentials. */
export const ACCOUNT_SCOPED_STORAGE_KEYS: readonly string[] = [
  // Hugging Face credential, and the legacy training blob that still embeds one.
  "unsloth_hf_token",
  "unsloth_hf_token_migration_v1",
  "unsloth_hf_token_backend_revision",
  "unsloth_training_config_v1",
  // Dictation transcripts, kept in full inside the voice settings store.
  "unsloth_voice_settings",
  // Display name, nickname and avatar, which the personalization sync treats as
  // migration input when the new account has no record yet.
  "unsloth_user_profile",
  // Absolute path of the last model loaded, which the startup auto-load reads.
  "unsloth.last-local-model-load.v1",
  // Hub search terms, which are repository and model names somebody typed.
  "unsloth.hub.recentSearches",
  // Custom provider names, base URLs and model lists, plus their key handles.
  "unsloth_chat_external_providers",
  "unsloth_chat_external_provider_keys",
  "unsloth_chat_connections_enabled",
  // System prompts and presets, which the legacy chat-settings import copies.
  "unsloth_chat_custom_presets",
  "unsloth_chat_active_preset",
  "unsloth_chat_active_preset_source",
  "unsloth_chat_system_prompts",
];

/** Key families, one entry per thread. */
export const ACCOUNT_SCOPED_STORAGE_PREFIXES: readonly string[] = [
  "chat-draft:",
  "chat-draft-pastes:",
];

function removeItem(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    // A blocked or full storage is not a reason to refuse the sign-in.
  }
}

function readItem(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeItem(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Same as above: best effort.
  }
}

/** Every key currently in localStorage, however this environment enumerates it. */
function storedKeys(): string[] {
  try {
    const storage = localStorage;
    if (typeof storage.length === "number" && typeof storage.key === "function") {
      const keys: string[] = [];
      for (let index = 0; index < storage.length; index += 1) {
        const key = storage.key(index);
        if (key !== null) keys.push(key);
      }
      return keys;
    }
    return Object.keys(storage);
  } catch {
    return [];
  }
}


/** Drop every account-scoped key, whole or prefixed. */
export function purgeAccountScopedBrowserState(): void {
  for (const key of ACCOUNT_SCOPED_STORAGE_KEYS) removeItem(key);
  for (const key of storedKeys()) {
    if (ACCOUNT_SCOPED_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
      removeItem(key);
    }
  }
}

/**
 * Whether this account may read the browser's pre-accounts data as its own.
 *
 * The legacy chat store, the legacy chat settings and the legacy Hugging Face
 * token were written by the single user of an install that had no accounts, so
 * they belong to whoever first signs in here after the upgrade. For anyone else
 * they are another person's content, and the import markers do not help: they
 * are origin-wide too, and the merge paths fall back to legacy rows whenever the
 * signing-in account's own list is empty.
 */
export function legacyBrowserDataBelongsToCurrentAccount(): boolean {
  const owner = readItem(LEGACY_DATA_OWNER_KEY);
  if (owner === null) {
    // Unclaimed. That is the ordinary upgrade: an install that had one user has
    // a live session already, and nobody signs in through the form again, so the
    // data is that user's and the migrations should run exactly as before. Once
    // any account has signed in here without claiming it, it is not theirs.
    return readItem(LAST_ACCOUNT_KEY) === null;
  }
  return owner === readItem(LAST_ACCOUNT_KEY);
}

/**
 * Record the account that just authenticated, purging if it is a different one.
 *
 * Returns whether a purge happened, which is what a caller uses to decide
 * whether it also has to reset in-memory stores before the new session mounts.
 */
export function notifyAccountAuthenticated(
  username: string,
  installationOwner?: string | null,
): boolean {
  const account = username.trim().toLowerCase();
  if (!account) return false;
  const previous = readItem(LAST_ACCOUNT_KEY);
  writeItem(LAST_ACCOUNT_KEY, account);
  // Whoever signed in first is not the owner of the pre-accounts data: on an
  // upgraded install a managed account can reach this page first, and claiming
  // ownership there hands it the original user's conversations, prompts and Hub
  // credential to migrate into its own workspace. The server says who the owner
  // is, so use that and leave the data unattributed until they sign in.
  const owner = (installationOwner ?? "").trim().toLowerCase();
  if (owner && account === owner) writeItem(LEGACY_DATA_OWNER_KEY, account);
  if (previous === null || previous === account) return false;
  purgeAccountScopedBrowserState();
  try {
    window.dispatchEvent(new Event(ACCOUNT_CHANGED_EVENT));
  } catch {
    // No window (tests, SSR): the stores that listen do not exist either.
  }
  return true;
}
