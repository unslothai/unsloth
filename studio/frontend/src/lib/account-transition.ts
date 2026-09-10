// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { USER_STOPPED_KEY } from "../hooks/server-stop-intent.ts";

export const BROWSER_ACCOUNT_KEY = "unsloth.browser-account.v1";
/** Written before a switch publishes new tokens, so peer tabs stop sending requests until the
 * marker lands and they reload. */
export const BROWSER_ACCOUNT_FENCE_KEY = "unsloth.browser-account.fence.v1";
/** A peer tab holds the marker for at most this long before reloading on its own. */
export const ACCOUNT_FENCE_TIMEOUT_MS = 10_000;

let transitionPending = false;
/** True in a peer tab between another tab's fence and its marker: requests must not go out. */
export function accountTransitionPending(): boolean {
  return transitionPending;
}
export const OWNER_BROWSER_ACCOUNT = "unsloth";

export const APPEARANCE_KEY = "unsloth_appearance_customization";

/** Browser chrome only. Never add credentials, content, model choices or profile data. */
export const ACCOUNT_CHROME_KEYS = new Set([
  "theme",
  "palette",
  APPEARANCE_KEY,
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
export const ACCOUNT_SESSION_CHROME_KEYS = new Set([USER_STOPPED_KEY]);
/** Purged on an account change. Durable stores are named per account (`accountDatabaseName`). */
export const ACCOUNT_DATABASES = [
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

/** The owner keeps the historical name; a managed account gets its own store. */
export function accountDatabaseName(
  name: string,
  storage: Pick<Storage, "getItem"> | null = typeof window === "undefined"
    ? null
    : window.localStorage,
): string {
  const account = parseAccountMarker(
    storage?.getItem(BROWSER_ACCOUNT_KEY) ?? OWNER_BROWSER_ACCOUNT,
  );
  const owner =
    account.accountId === "owner" ||
    (!account.accountId && account.username === OWNER_BROWSER_ACCOUNT);
  return owner ? name : `${name}:${account.accountId ?? account.username}`;
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

/** May the browser data carry over? The username fallback cannot tell a recreated account apart. */
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

const IMPORTED_FONT_SELECTIONS = [
  "uiFont",
  "headingFont",
  "chatFont",
  "codeFont",
] as const;

/** An imported font is uploaded file bytes, not chrome: strip it and any selection naming it. */
function purgeImportedFonts(storage: Storage): void {
  const raw = storage.getItem(APPEARANCE_KEY);
  if (!raw?.includes("importedFonts")) return;
  let parsed: {
    state?: { customization?: Record<string, unknown> };
    customization?: Record<string, unknown>;
  };
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Unreadable, so worthless to the store, and it still holds font bytes.
    storage.removeItem(APPEARANCE_KEY);
    return;
  }
  const customization = parsed?.state?.customization ?? parsed?.customization;
  const fonts = customization?.importedFonts;
  if (!customization || !Array.isArray(fonts) || fonts.length === 0) return;
  const names = new Set(
    fonts.map((font) => (font as { name?: unknown } | null)?.name),
  );
  customization.importedFonts = [];
  for (const field of IMPORTED_FONT_SELECTIONS) {
    if (names.has(customization[field])) customization[field] = null;
  }
  storage.setItem(APPEARANCE_KEY, JSON.stringify(parsed));
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
    // Unreadable session storage never fails a sign-in.
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

/** Run before publishing new tokens; the marker is published last so other tabs reload only
 * once the new session is ready. */
export async function transitionBrowserAccount(
  account: BrowserAccount | string,
  postAuthRoute: string,
  commitSession: () => void,
  browser: AccountTransitionBrowser = window,
): Promise<boolean> {
  const marker = browserAccountMarker(account);
  const storage = browser.localStorage;
  const changed = !isSameAccount(
    parseAccountMarker(
      storage.getItem(BROWSER_ACCOUNT_KEY) ?? OWNER_BROWSER_ACCOUNT,
    ),
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
    purgeImportedFonts(storage);
    clearAccountSessionStorage(browser);
    await Promise.all(
      ACCOUNT_DATABASES.map((name) =>
        deleteAccountDatabase(browser.indexedDB, name),
      ),
    );
  }
  // Fence first: peers see it before the tokens, so nothing of the previous account goes out
  // under the new credentials while their reload is pending.
  if (changed) storage.setItem(BROWSER_ACCOUNT_FENCE_KEY, marker);
  commitSession();
  if (storage.getItem(BROWSER_ACCOUNT_KEY) !== marker)
    storage.setItem(BROWSER_ACCOUNT_KEY, marker);
  if (changed) storage.removeItem(BROWSER_ACCOUNT_FENCE_KEY);
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
  let fenceTimer: ReturnType<typeof setTimeout> | null = null;
  const reload = () => {
    if (reloading) return;
    reloading = true;
    if (fenceTimer !== null) clearTimeout(fenceTimer);
    clearAccountSessionStorage(browser);
    browser.location.reload();
  };
  browser.addEventListener("storage", (event) => {
    if (reloading || event.newValue === null) return;
    if (event.storageArea && event.storageArea !== browser.localStorage) return;
    if (event.key === BROWSER_ACCOUNT_FENCE_KEY) {
      const current = parseAccountMarker(
        browser.localStorage.getItem(BROWSER_ACCOUNT_KEY) ?? OWNER_BROWSER_ACCOUNT,
      );
      if (isSameAccount(current, parseAccountMarker(event.newValue))) return;
      // Stop sending until the marker arrives; a switch that never finishes still reloads.
      transitionPending = true;
      if (fenceTimer === null) fenceTimer = setTimeout(reload, ACCOUNT_FENCE_TIMEOUT_MS);
      return;
    }
    if (event.key !== BROWSER_ACCOUNT_KEY) return;
    const previous = parseAccountMarker(
      event.oldValue ?? OWNER_BROWSER_ACCOUNT,
    );
    if (isSameAccount(previous, parseAccountMarker(event.newValue))) return;
    reload();
  });
}
