// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const BROWSER_ACCOUNT_KEY = "unsloth.browser-account.v1";

/** Browser chrome only. Never add credentials, content, model choices or profile data. */
export const ACCOUNT_CHROME_KEYS = new Set([
  "theme",
  "palette",
  "unsloth_appearance_customization",
  "unsloth_locale",
  "sidebar_pinned",
  "sidebar_width",
  "chat_settings_width",
  "unsloth_sidebar_navigate_open",
  "unsloth_settings_active_tab",
  "unsloth_loaded_models_collapsed",
  "unsloth_loaded_models_dismissed",
  "unsloth-rag-preview-width",
]);
// Version-specific notice dismissals contain no account data.
export const ACCOUNT_CHROME_PREFIXES = ["unsloth_web_update_dismissed:"] as const;
export const ACCOUNT_DATABASES = [
  "unsloth-data-recipes",
  "unsloth-data-recipe-executions",
] as const;

export type AccountTransitionBrowser = Pick<Window, "localStorage" | "indexedDB" | "location">;

/** Account names accepted by the account API are ASCII; case folding is locale independent. */
export function normalizeAccountUsername(username: string): string {
  return username.trim().toLowerCase();
}

export function resetFullAccessForMultiUser(storage: Storage): void {
  if (storage.getItem("unsloth_chat_permission_mode") === "full") {
    storage.setItem("unsloth_chat_permission_mode", "auto");
  }
}

function deleteAccountDatabase(indexedDB: IDBFactory, name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error ?? new Error("Could not clear account data."));
    request.onblocked = () => reject(new Error(
      "Close other Unsloth tabs and retry signing in to clear the previous account's data.",
    ));
  });
}

/**
 * Run before publishing new tokens. An absent marker denotes the historical owner browser.
 * Publish the marker last so other tabs reload only after the new session is ready.
 * Returns true when a document navigation replaces every hydrated store.
 */
export async function transitionBrowserAccount(
  username: string,
  postAuthRoute: string,
  commitSession: () => void,
  browser: AccountTransitionBrowser = window,
): Promise<boolean> {
  const next = normalizeAccountUsername(username);
  if (!next) throw new Error("Missing account username.");
  const storage = browser.localStorage;
  const previous = normalizeAccountUsername(storage.getItem(BROWSER_ACCOUNT_KEY) ?? "unsloth");
  const changed = previous !== next;
  if (changed) {
    const keys = Array.from({ length: storage.length }, (_, index) => storage.key(index));
    for (const key of keys) {
      if (!key || key === BROWSER_ACCOUNT_KEY || ACCOUNT_CHROME_KEYS.has(key) ||
          ACCOUNT_CHROME_PREFIXES.some((prefix) => key.startsWith(prefix))) continue;
      if (key.startsWith("unsloth") || key.startsWith("chat-draft")) storage.removeItem(key);
    }
    await Promise.all(ACCOUNT_DATABASES.map((name) => deleteAccountDatabase(browser.indexedDB, name)));
  }
  commitSession();
  if (storage.getItem(BROWSER_ACCOUNT_KEY) !== next) storage.setItem(BROWSER_ACCOUNT_KEY, next);
  if (changed) browser.location.replace(postAuthRoute);
  return changed;
}

/** One listener and at most one reload per document, including duplicate storage events. */
const watchedBrowsers = new WeakSet<Window>();
export function installAccountTransitionListener(browser: Window = window): void {
  if (watchedBrowsers.has(browser)) return;
  watchedBrowsers.add(browser);
  let reloading = false;
  browser.addEventListener("storage", (event) => {
    if (reloading || event.key !== BROWSER_ACCOUNT_KEY || event.newValue === null) return;
    if (event.storageArea && event.storageArea !== browser.localStorage) return;
    const previous = normalizeAccountUsername(event.oldValue ?? "unsloth");
    if (previous === normalizeAccountUsername(event.newValue)) return;
    reloading = true;
    browser.location.reload();
  });
}
