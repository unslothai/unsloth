// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import {
  LOCALES,
  type Locale,
  isSupportedLocale,
  loadLocaleMessages,
} from "./messages";

export const DEFAULT_LOCALE: Locale = "en";
export const AUTO_LOCALE = "auto";
export const LOCALE_STORAGE_KEY = "unsloth_locale";
export const LOCALE_INITIALIZATION_TIMEOUT_MS = 2_000;

export type LocalePreference = Locale | typeof AUTO_LOCALE;

export type LocaleChangeResult =
  | "applied"
  | "superseded"
  | "failed"
  | "cancelled";

export type SetLocaleOptions = {
  loadMessages?: (locale: Locale) => Promise<void> | undefined;
  signal?: AbortSignal;
  /**
   * Adopt the preference even when its catalog never loads, rendering English.
   *
   * For a user picking a language this would be wrong: the choice failed, so it
   * must not be persisted. Hydration is the opposite case. The preference it is
   * applying is already the stored truth on the server, so refusing to adopt it
   * leaves the local preference disagreeing with the server, and the next
   * outbound save would push the stale local value back over it.
   */
  adoptOnFailure?: boolean;
};

export const DEFAULT_LOCALE_PREFERENCE: LocalePreference = AUTO_LOCALE;

const subscribers = new Set<() => void>();

let currentPreference: LocalePreference = DEFAULT_LOCALE_PREFERENCE;
let currentLocale: Locale = DEFAULT_LOCALE;
let pendingPreference: LocalePreference | null = null;
let pendingPreferenceShouldPersist = false;
let areListenersActive = false;

export function isLocalePreference(value: unknown): value is LocalePreference {
  return value === AUTO_LOCALE || isSupportedLocale(value);
}

function isTraditionalChinese(lowerTag: string): boolean {
  if (lowerTag.includes("hant")) return true;
  if (lowerTag.includes("hans")) return false;
  const parts = lowerTag.split("-");
  return parts.includes("tw") || parts.includes("hk") || parts.includes("mo");
}

function matchLocale(tag: string): Locale | null {
  const locales = Object.keys(LOCALES) as Locale[];
  const lower = tag.toLowerCase();
  const exact = locales.find((locale) => locale.toLowerCase() === lower);
  if (exact) return exact;
  const language = lower.split("-")[0];
  // We only ship Simplified Chinese. Don't hand it to Traditional Chinese
  // (zh-Hant / zh-TW / zh-HK / zh-MO) users; let them fall through instead.
  if (language === "zh" && isTraditionalChinese(lower)) return null;
  return (
    locales.find((locale) => locale.toLowerCase().split("-")[0] === language) ??
    null
  );
}

function detectLocale(): Locale {
  const navigatorRef = globalThis.navigator;
  const tags = navigatorRef?.languages?.length
    ? navigatorRef.languages
    : navigatorRef?.language
      ? [navigatorRef.language]
      : [];
  let sawTraditionalChinese = false;
  for (const tag of tags) {
    const lower = tag.toLowerCase();
    if (lower.split("-")[0] === "zh" && isTraditionalChinese(lower)) {
      sawTraditionalChinese = true;
    } else if (sawTraditionalChinese && lower === "zh") {
      // Bare zh after a Traditional tag is the browser's base-subtag fallback, not a Simplified request; skip it.
      continue;
    }
    const match = matchLocale(tag);
    if (match) return match;
  }
  return DEFAULT_LOCALE;
}

function normalizePreference(value: unknown): LocalePreference {
  if (value === AUTO_LOCALE) return AUTO_LOCALE;
  // Return a value re-derived from our own locale table rather than the raw
  // input, so only known language codes are ever persisted.
  const locales = Object.keys(LOCALES) as Locale[];
  return (
    locales.find((locale) => locale === value) ?? DEFAULT_LOCALE_PREFERENCE
  );
}

function resolvePreference(preference: LocalePreference): Locale {
  return preference === AUTO_LOCALE ? detectLocale() : preference;
}

function readStoredPreference(): LocalePreference {
  try {
    const stored = globalThis.localStorage?.getItem(LOCALE_STORAGE_KEY) ?? null;
    return normalizePreference(stored);
  } catch {
    return DEFAULT_LOCALE_PREFERENCE;
  }
}

function writeStoredPreference(preference: LocalePreference): void {
  try {
    globalThis.localStorage?.setItem(LOCALE_STORAGE_KEY, preference);
  } catch {
    // localStorage can be disabled; a failure only costs persistence, not
    // the language of the current session.
  }
}

function syncDocumentLang(locale: Locale): void {
  if (typeof document === "undefined") return;
  document.documentElement.lang = locale;
}

function notifySubscribers(): void {
  for (const subscriber of subscribers) subscriber();
}

let preferenceRevision = 0;

function commitPreference(
  preference: LocalePreference,
  locale: Locale,
  revision: number,
  persist: boolean,
): LocaleChangeResult {
  if (revision !== preferenceRevision) return "superseded";
  if (persist) writeStoredPreference(preference);
  const didChange =
    preference !== currentPreference ||
    locale !== currentLocale ||
    pendingPreference !== null;
  currentPreference = preference;
  currentLocale = locale;
  pendingPreference = null;
  pendingPreferenceShouldPersist = false;
  syncDocumentLang(locale);
  if (didChange) notifySubscribers();
  return "applied";
}

function commitFallbackLocale(
  preference: LocalePreference,
  revision: number,
): void {
  if (revision !== preferenceRevision) return;
  const didChange =
    preference !== currentPreference ||
    currentLocale !== DEFAULT_LOCALE ||
    pendingPreference !== null;
  currentPreference = preference;
  currentLocale = DEFAULT_LOCALE;
  pendingPreference = null;
  pendingPreferenceShouldPersist = false;
  syncDocumentLang(currentLocale);
  if (didChange) notifySubscribers();
}

type LocaleCatalogLoader = (locale: Locale) => Promise<void> | undefined;

function failPreference(revision: number): void {
  if (revision !== preferenceRevision || pendingPreference === null) return;
  pendingPreference = null;
  pendingPreferenceShouldPersist = false;
  notifySubscribers();
}

function cancelPreference(revision: number): void {
  if (revision !== preferenceRevision) return;
  preferenceRevision += 1;
  if (pendingPreference === null) return;
  pendingPreference = null;
  pendingPreferenceShouldPersist = false;
  notifySubscribers();
}

function applyPreference(
  preference: LocalePreference,
  persist = false,
  loadMessages: LocaleCatalogLoader = loadLocaleMessages,
  signal?: AbortSignal,
  adoptOnFailure = false,
): LocaleChangeResult | Promise<LocaleChangeResult> {
  if (signal?.aborted) return "cancelled";
  const previousPending = pendingPreference;
  const previousPendingPersist = pendingPreferenceShouldPersist;
  const revision = ++preferenceRevision;
  const locale = resolvePreference(preference);
  let pending: Promise<void> | undefined;
  try {
    pending = loadMessages(locale);
  } catch {
    if (adoptOnFailure) {
      commitFallbackLocale(preference, revision);
      return "failed";
    }
    // This request never became the pending one, so failing it must not clear
    // the marker an earlier request that is still in flight is relying on.
    pendingPreference = previousPending;
    pendingPreferenceShouldPersist = previousPendingPersist;
    return "failed";
  }
  if (!pending) {
    return commitPreference(preference, locale, revision, persist);
  }
  pendingPreference = preference;
  pendingPreferenceShouldPersist = persist;
  notifySubscribers();
  const cancel = () => cancelPreference(revision);
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  return pending
    .then<LocaleChangeResult, LocaleChangeResult>(
      () => {
        if (signal?.aborted) return "cancelled";
        return commitPreference(preference, locale, revision, persist);
      },
      () => {
        if (signal?.aborted) return "cancelled";
        if (revision !== preferenceRevision) return "superseded";
        if (adoptOnFailure) {
          commitFallbackLocale(preference, revision);
        } else {
          failPreference(revision);
        }
        return "failed";
      },
    )
    .finally(() => signal?.removeEventListener("abort", cancel));
}

function isLocaleStorageEvent(event: StorageEvent): boolean {
  if (event.key !== LOCALE_STORAGE_KEY && event.key !== null) return false;
  if (!event.storageArea || typeof window === "undefined") return true;
  // Accessing window.localStorage can throw in privacy-restricted contexts
  // where storage is blocked; mirror the try/catch in readStoredPreference/
  // writeStoredPreference so storage-event handling is just as resilient.
  try {
    return event.storageArea === window.localStorage;
  } catch {
    return false;
  }
}

function handleStorageEvent(event: StorageEvent): void {
  if (!isLocaleStorageEvent(event)) return;
  const nextPreference =
    event.key === null
      ? DEFAULT_LOCALE_PREFERENCE
      : normalizePreference(event.newValue);
  void applyPreference(nextPreference);
}

function handleLanguageChange(): void {
  // A pending choice is the effective preference. Browser changes may refresh
  // a pending auto request, but must not supersede an explicit user choice.
  const effectivePreference = pendingPreference ?? currentPreference;
  if (effectivePreference !== AUTO_LOCALE) return;
  void applyPreference(effectivePreference, pendingPreferenceShouldPersist);
}

function startListeners(): void {
  if (areListenersActive || typeof window === "undefined") return;
  window.addEventListener("storage", handleStorageEvent);
  window.addEventListener("languagechange", handleLanguageChange);
  areListenersActive = true;
}

function stopListeners(): void {
  if (!areListenersActive || typeof window === "undefined") return;
  window.removeEventListener("storage", handleStorageEvent);
  window.removeEventListener("languagechange", handleLanguageChange);
  areListenersActive = false;
}

function getLocaleSnapshot(): Locale {
  return currentLocale;
}

function getServerLocaleSnapshot(): Locale {
  return DEFAULT_LOCALE;
}

function getPreferenceSnapshot(): LocalePreference {
  return currentPreference;
}

function getServerPreferenceSnapshot(): LocalePreference {
  return DEFAULT_LOCALE_PREFERENCE;
}

function getPendingPreferenceSnapshot(): LocalePreference | null {
  return pendingPreference;
}

function getServerPendingPreferenceSnapshot(): null {
  return null;
}

export function subscribeLocale(listener: () => void): () => void {
  const shouldStartListeners = subscribers.size === 0;
  subscribers.add(listener);
  if (shouldStartListeners) startListeners();

  return () => {
    subscribers.delete(listener);
    if (subscribers.size === 0) stopListeners();
  };
}

export function initializeLocale({
  loadMessages = loadLocaleMessages,
  timeoutMs = LOCALE_INITIALIZATION_TIMEOUT_MS,
}: {
  loadMessages?: LocaleCatalogLoader;
  timeoutMs?: number;
} = {}): Locale | Promise<Locale> {
  const preference = readStoredPreference();
  const locale = resolvePreference(preference);
  const revision = ++preferenceRevision;
  let pending: Promise<void> | undefined;
  try {
    pending = loadMessages(locale);
  } catch {
    commitFallbackLocale(preference, revision);
    return currentLocale;
  }
  if (!pending) {
    commitPreference(preference, locale, revision, false);
    return currentLocale;
  }

  pendingPreference = preference;
  pendingPreferenceShouldPersist = false;
  notifySubscribers();

  return new Promise((resolve) => {
    let didResolve = false;
    const finish = () => {
      if (didResolve) return;
      didResolve = true;
      resolve(currentLocale);
    };
    const timeout = globalThis.setTimeout(
      () => {
        commitFallbackLocale(preference, revision);
        finish();
      },
      Math.max(0, timeoutMs),
    );

    pending.then(
      () => {
        globalThis.clearTimeout(timeout);
        commitPreference(preference, locale, revision, false);
        finish();
      },
      () => {
        globalThis.clearTimeout(timeout);
        commitFallbackLocale(preference, revision);
        finish();
      },
    );
  });
}

export function getLocale(): Locale {
  return currentLocale;
}

export function getLocalePreference(): LocalePreference {
  return currentPreference;
}

export function getPendingLocalePreference(): LocalePreference | null {
  return pendingPreference;
}

export function setLocale(
  preference: LocalePreference,
  {
    loadMessages = loadLocaleMessages,
    signal,
    adoptOnFailure = false,
  }: SetLocaleOptions = {},
): LocaleChangeResult | Promise<LocaleChangeResult> {
  const requestedPreference = normalizePreference(preference);
  // persist stays true on both paths: a successful change is still written, and
  // the adopt-on-failure path routes through commitFallbackLocale, which never
  // writes storage, so a preference whose catalog failed is adopted for this
  // session without being recorded as a choice that worked.
  return applyPreference(
    requestedPreference, true, loadMessages, signal, adoptOnFailure,
  );
}

export function useLocale(): Locale {
  return useSyncExternalStore(
    subscribeLocale,
    getLocaleSnapshot,
    getServerLocaleSnapshot,
  );
}

export function useLocalePreference(): LocalePreference {
  return useSyncExternalStore(
    subscribeLocale,
    getPreferenceSnapshot,
    getServerPreferenceSnapshot,
  );
}

export function usePendingLocalePreference(): LocalePreference | null {
  return useSyncExternalStore(
    subscribeLocale,
    getPendingPreferenceSnapshot,
    getServerPendingPreferenceSnapshot,
  );
}
