// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { USER_STOPPED_KEY } from "../hooks/server-stop-intent.ts";

export const BROWSER_ACCOUNT_KEY = "unsloth.browser-account.v1";
export const OWNER_BROWSER_ACCOUNT = "unsloth";

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
export const ACCOUNT_CHROME_PREFIXES = [
  "unsloth_web_update_dismissed:",
] as const;
/** Per-tab flags about the browser session, not the account. Never add content. */
export const ACCOUNT_SESSION_CHROME_KEYS = new Set([
  USER_STOPPED_KEY,
]);
export const ACCOUNT_DATABASES = [
  "unsloth-data-recipes",
  "unsloth-data-recipe-executions",
  // Legacy store: its one-shot import would push these threads into the next account.
  "unsloth-chat",
] as const;

export type AccountTransitionBrowser = Pick<
  Window,
  "localStorage" | "sessionStorage" | "indexedDB" | "location"
>;

export type BrowserAccount = { username: string; accountId?: string | null };

const ACCOUNT_ID_MARKER_PREFIX = "account:";

/** Must match what `auth/storage.py` stores: casefolded `[a-z0-9_-]{3,32}`. */
export function normalizeAccountUsername(username: string): string {
  return username.trim().toLowerCase();
}

export function browserAccountMarker(account: BrowserAccount | string): string {
  const identity: BrowserAccount =
    typeof account === "string" ? { username: account } : account;
  const username = normalizeAccountUsername(identity.username);
  if (!username) throw new Error("Missing account username.");
  return identity.accountId
    ? `${ACCOUNT_ID_MARKER_PREFIX}${identity.accountId}:${username}`
    : username;
}

type MarkedAccount = { accountId: string | null; username: string };
function parseAccountMarker(marker: string): MarkedAccount {
  const legacy = {
    accountId: null,
    username: normalizeAccountUsername(marker),
  };
  if (!marker.startsWith(ACCOUNT_ID_MARKER_PREFIX)) return legacy;
  const qualified = marker.slice(ACCOUNT_ID_MARKER_PREFIX.length);
  const separator = qualified.indexOf(":");
  if (separator <= 0 || separator === qualified.length - 1) return legacy;
  return {
    accountId: qualified.slice(0, separator),
    username: normalizeAccountUsername(qualified.slice(separator + 1)),
  };
}

/** Whether the browser's data may carry over. Ids decide when both sides have one; the username fallback cannot tell a recreated account apart. */
function isSameAccount(previous: MarkedAccount, next: MarkedAccount): boolean {
  if (previous.accountId && next.accountId)
    return previous.accountId === next.accountId;
  return previous.username === next.username;
}

export function resetFullAccessForMultiUser(storage: Storage): void {
  if (storage.getItem("unsloth_chat_permission_mode") === "full") {
    storage.setItem("unsloth_chat_permission_mode", "auto");
  }
}

function clearAccountSessionStorage(browser: AccountTransitionBrowser): void {
  try {
    const storage = browser.sessionStorage;
    const keys = Array.from({ length: storage.length }, (_, index) =>
      storage.key(index),
    );
    for (const key of keys) {
      if (!key || ACCOUNT_SESSION_CHROME_KEYS.has(key)) continue;
      storage.removeItem(key);
    }
  } catch {
  }
}

function deleteAccountDatabase(
  indexedDB: IDBFactory,
  name: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onerror = () =>
      reject(request.error ?? new Error("Could not clear account data."));
    request.onblocked = () =>
      reject(
        new Error(
          "Close other Unsloth tabs and retry signing in to clear the previous account's data.",
        ),
      );
  });
}

/** Run before publishing new tokens; an absent marker means the historical owner browser, and the marker is published last so other tabs reload only once the new session is ready. */
export async function transitionBrowserAccount(
  account: BrowserAccount | string,
  postAuthRoute: string,
  commitSession: () => void,
  browser: AccountTransitionBrowser = window,
): Promise<boolean> {
  const marker = browserAccountMarker(account);
  const storage = browser.localStorage;
  const changed = !isSameAccount(
    parseAccountMarker(storage.getItem(BROWSER_ACCOUNT_KEY) ?? OWNER_BROWSER_ACCOUNT),
    parseAccountMarker(marker),
  );
  if (changed) {
    const keys = Array.from({ length: storage.length }, (_, index) =>
      storage.key(index),
    );
    for (const key of keys) {
      if (
        !key ||
        key === BROWSER_ACCOUNT_KEY ||
        ACCOUNT_CHROME_KEYS.has(key) ||
        ACCOUNT_CHROME_PREFIXES.some((prefix) => key.startsWith(prefix))
      )
        continue;
      if (key.startsWith("unsloth") || key.startsWith("chat-draft"))
        storage.removeItem(key);
    }
    clearAccountSessionStorage(browser);
    await Promise.all(
      ACCOUNT_DATABASES.map((name) =>
        deleteAccountDatabase(browser.indexedDB, name),
      ),
    );
  }
  commitSession();
  if (storage.getItem(BROWSER_ACCOUNT_KEY) !== marker)
    storage.setItem(BROWSER_ACCOUNT_KEY, marker);
  if (changed) browser.location.replace(postAuthRoute);
  return changed;
}

const watchedBrowsers = new WeakSet<Window>();
export function installAccountTransitionListener(
  browser: Window = window,
): void {
  if (watchedBrowsers.has(browser)) return;
  watchedBrowsers.add(browser);
  let reloading = false;
  browser.addEventListener("storage", (event) => {
    if (
      reloading ||
      event.key !== BROWSER_ACCOUNT_KEY ||
      event.newValue === null
    )
      return;
    if (event.storageArea && event.storageArea !== browser.localStorage) return;
    const previous = parseAccountMarker(event.oldValue ?? OWNER_BROWSER_ACCOUNT);
    if (isSameAccount(previous, parseAccountMarker(event.newValue))) return;
    reloading = true;
    browser.location.reload();
  });
}
