// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Browser-local state that belongs to one account, and what to do when another
 * one signs in on the same browser.
 *
 * localStorage and IndexedDB are origin-wide and survive a logout, which was
 * correct while there was one user. Now a value left behind is read by the next
 * account, or worse treated as migration input and copied into its workspace.
 *
 * Two rules, at the one point a username is admitted: content keys are cleared
 * when the account changes, and legacy migration input is bound to whoever owns
 * this browser's pre-accounts data.
 */

const LAST_ACCOUNT_KEY = "unsloth.browser-account.v1";
const LEGACY_DATA_OWNER_KEY = "unsloth.legacy-data-owner.v1";
const LEGACY_QUARANTINE_KEY = "unsloth.legacy-quarantine.v1";

/**
 * Dispatched on window when a different account signs in on this browser.
 *
 * Clearing a key does not change an already-hydrated store: a same-window
 * removeItem fires no storage event and the SPA does not reload. Stores holding
 * account content listen for this and reset themselves.
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
  // Display name and avatar, which the personalization sync treats as migration
  // input when the new account has no record yet.
  "unsloth_user_profile",
  // Absolute path of the last model loaded, which the startup auto-load reads.
  "unsloth.last-local-model-load.v1",
  // Delete confirmations, which decide what the next account's first delete does.
  "unsloth_chat_preferences",
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


/**
 * Keys that are chrome rather than content, and so survive an account change.
 *
 * The list below is INVERTED on purpose. An allowlist of account-scoped keys
 * missed four stores in a single review round (downloads, per-model configs,
 * recipe databases, pending chat media), and each miss is one account's data
 * read by the next. The app stores over a hundred keys and grows more; a new one
 * is cleared until somebody decides it is chrome, because that failure is a
 * re-run of a preference and the other one is a disclosure.
 */
const ACCOUNT_NEUTRAL_STORAGE_KEYS: ReadonlySet<string> = new Set([
  // The session itself, and this module's own bookkeeping.
  "unsloth_auth_token",
  LAST_ACCOUNT_KEY,
  LEGACY_DATA_OWNER_KEY,
  LEGACY_QUARANTINE_KEY,
  // Appearance, language and layout.
  "unsloth_appearance_customization",
  "unsloth_locale",
  "unsloth-rag-preview-width",
  "unsloth_loaded_models_collapsed",
  "unsloth_loaded_models_dismissed",
  "unsloth_loaded_models_position",
  "unsloth_show_loaded_models_indicator",
  "unsloth_model_selector_section",
  "unsloth_settings_active_tab",
  "unsloth_settings_panel_prefs",
  "unsloth_sidebar_navigate_open",
  "unsloth_sidebar_organization",
  "unsloth_monitor_overlay",
  "unsloth_api_monitor_overlay",
  "unsloth_video_advanced_open",
  "unsloth_plus_menu_pins",
  // Which tab or view was open, and filters over a shared catalog.
  "unsloth.hub.allModelsView",
  "unsloth.hub.inventorySort",
  "unsloth.hub.modelsTab",
  "unsloth.hub.ownerScope",
  "unsloth.studio.train.datasetPickerTab",
  "unsloth.studio.train.modelPickerTab",
  "unsloth.studio.train.paramMode",
  "unsloth_train_param_mode",
  "unsloth_models_fit_on_device_only",
  // Notices already dismissed, and install-level transport.
  "unsloth_onboarding_done",
  "unsloth_web_update_dismissed",
  "unsloth_show_llama_update_banner",
  "unsloth.studio.xetNoticeCount",
  "unsloth.studio.xetNoticeMigrated",
  "unsloth.studio.transportMode",
]);

/** Everything this app persists starts with one of these. */
const APP_STORAGE_PREFIXES: readonly string[] = ["unsloth_", "unsloth.", "unsloth-", "chat-draft"];

/** Whether a stored key holds content belonging to one account. */
function isAccountScopedKey(key: string): boolean {
  if (ACCOUNT_NEUTRAL_STORAGE_KEYS.has(key)) return false;
  if (ACCOUNT_SCOPED_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) return true;
  // Only this app's own keys: a third-party library's storage is not ours to delete.
  return APP_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix));
}

/**
 * IndexedDB databases holding one account's records.
 *
 * localStorage was only ever half of it: recipes carry plaintext provider and
 * Hub tokens, prompts and local paths, and execution records carry generated
 * rows and logs.
 */
const ACCOUNT_SCOPED_DATABASES: readonly string[] = [
  "unsloth-data-recipes",
  "unsloth-data-recipe-executions",
];

function deleteAccountScopedDatabases(): void {
  try {
    if (typeof indexedDB === "undefined") return;
    for (const name of ACCOUNT_SCOPED_DATABASES) {
      try {
        indexedDB.deleteDatabase(name);
      } catch {
        // An open handle blocks the delete; the reload below drops it, and the
        // next transition deletes it for real.
      }
    }
  } catch {
    // No IndexedDB (tests, private modes that disable it): nothing to drop.
  }
}

/** Drop every account-scoped key and database. */
export function purgeAccountScopedBrowserState(): void {
  for (const key of ACCOUNT_SCOPED_STORAGE_KEYS) removeItem(key);
  for (const key of storedKeys()) {
    if (isAccountScopedKey(key)) removeItem(key);
  }
  deleteAccountScopedDatabases();
}

/**
 * Whether this account may read the browser's pre-accounts data as its own.
 *
 * The legacy chat store, chat settings and Hugging Face token were written by
 * the single user of an install that had no accounts. The import markers do not
 * help: they are origin-wide too, and the merge paths fall back to legacy rows
 * whenever the signing-in account's own list is empty.
 */
export function legacyBrowserDataBelongsToCurrentAccount(): boolean {
  const owner = readItem(LEGACY_DATA_OWNER_KEY);
  if (owner === null) {
    // Unclaimed is the ordinary upgrade: the single user has a live session and
    // never signs in again, so the migrations run as before. Once any account has
    // signed in here without claiming it, it is not theirs.
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
/**
 * Move the pre-accounts values aside, for a managed account signing in first.
 *
 * Not a purge: the owner marker gates only the three legacy migration helpers,
 * while the training, voice and profile stores hydrate straight from the keys.
 * Held until the owner signs in, rather than lost to whoever logged in first.
 */
function quarantineLegacyState(): void {
  const held: Record<string, string> = {};
  for (const key of ACCOUNT_SCOPED_STORAGE_KEYS) {
    const value = readItem(key);
    if (value !== null) held[key] = value;
  }
  for (const key of storedKeys()) {
    if (!ACCOUNT_SCOPED_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) continue;
    const value = readItem(key);
    if (value !== null) held[key] = value;
  }
  if (Object.keys(held).length === 0) return;
  try {
    writeItem(LEGACY_QUARANTINE_KEY, JSON.stringify(held));
  } catch {
    // Unserialisable or over quota: clearing without holding is still correct,
    // because the alternative is leaving it readable by this account.
  }
  purgeAccountScopedBrowserState();
}

/** Give the held values back to the account they belong to. */
function releaseLegacyQuarantine(): void {
  const raw = readItem(LEGACY_QUARANTINE_KEY);
  if (raw === null) return;
  removeItem(LEGACY_QUARANTINE_KEY);
  let held: Record<string, unknown>;
  try {
    held = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return;
  }
  for (const [key, value] of Object.entries(held)) {
    // Never over something this account has written since.
    if (typeof value === "string" && readItem(key) === null) writeItem(key, value);
  }
}

export function notifyAccountAuthenticated(
  username: string,
  installationOwner?: string | null,
): boolean {
  const account = username.trim().toLowerCase();
  if (!account) return false;
  const previous = readItem(LAST_ACCOUNT_KEY);
  writeItem(LAST_ACCOUNT_KEY, account);
  // Whoever signs in first is not the owner: on an upgraded install a managed
  // account can reach this page first, and claiming there hands it the original
  // user's conversations and Hub credential. The server says who the owner is.
  const owner = (installationOwner ?? "").trim().toLowerCase();
  const isOwner = Boolean(owner) && account === owner;
  if (isOwner) writeItem(LEGACY_DATA_OWNER_KEY, account);
  if (previous === null && !isOwner) {
    renderedAccount = account;
    // First sign-in after the upgrade, and not by the owner: everything on this
    // browser is the previous single user's.
    quarantineLegacyState();
    announceAccountChange();
    return true;
  }
  const changed = previous !== null && previous !== account;
  renderedAccount = account;
  if (changed) purgeAccountScopedBrowserState();
  // After the purge, never before it: the held values are this account's own and
  // clearing the incoming account's keys would take them straight back out.
  if (isOwner) releaseLegacyQuarantine();
  if (!changed) return false;
  announceAccountChange();
  return true;
}

function announceAccountChange(): void {
  try {
    window.dispatchEvent(new Event(ACCOUNT_CHANGED_EVENT));
  } catch {
    // No window (tests, SSR): the stores that listen do not exist either.
  }
}

// The account this document was rendered for, captured at load and updated by a
// sign-in in THIS tab. Compared against, rather than against localStorage: a
// storage event is delivered after the write, so reading the key back always
// returned the new value and the guard below never fired once.
let renderedAccount: string | null = null;
try {
  renderedAccount = readItem(LAST_ACCOUNT_KEY);
} catch {
  renderedAccount = null;
}

// A sign-in in one tab writes origin-wide storage every other tab's authFetch
// reads, so a tab mounted for the previous account sends its actions on the new
// token. The storage event is the only cross-document signal, and a reload is
// the one reset that covers every store rather than the listed few.
if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("storage", (event) => {
    const changed = (event as StorageEvent).key;
    if (changed !== LAST_ACCOUNT_KEY) return;
    const next = (event as StorageEvent).newValue;
    if (next === null || next === renderedAccount) return;
    renderedAccount = next;
    try {
      window.location.reload();
    } catch {
      // A window without navigation is a test harness, which has no UI to stale.
    }
  });
}
