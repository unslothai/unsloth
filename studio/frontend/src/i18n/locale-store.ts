// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import { isSupportedLocale, LOCALES, type Locale } from "./messages";

export const DEFAULT_LOCALE: Locale = "en";
export const AUTO_LOCALE = "auto";
export const LOCALE_STORAGE_KEY = "unsloth_locale";

export type LocalePreference = Locale | typeof AUTO_LOCALE;

export const DEFAULT_LOCALE_PREFERENCE: LocalePreference = AUTO_LOCALE;

const subscribers = new Set<() => void>();

let currentPreference: LocalePreference = DEFAULT_LOCALE_PREFERENCE;
let currentLocale: Locale = DEFAULT_LOCALE;
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
  for (const tag of tags) {
    const match = matchLocale(tag);
    if (match) return match;
  }
  return DEFAULT_LOCALE;
}

function normalizePreference(value: unknown): LocalePreference {
  return isLocalePreference(value) ? value : DEFAULT_LOCALE_PREFERENCE;
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
    // localStorage 可能被禁用；失败只影响持久化，不影响当前会话语言。
  }
}

function syncDocumentLang(locale: Locale): void {
  if (typeof document === "undefined") return;
  document.documentElement.lang = locale;
  document.documentElement.dir = LOCALES[locale].dir;
}

function notifySubscribers(): void {
  for (const subscriber of subscribers) subscriber();
}

function applyPreference(preference: LocalePreference): void {
  const locale = resolvePreference(preference);
  if (preference === currentPreference && locale === currentLocale) return;
  currentPreference = preference;
  currentLocale = locale;
  syncDocumentLang(locale);
  notifySubscribers();
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
  applyPreference(nextPreference);
}

function handleLanguageChange(): void {
  // Only auto mode tracks the browser language.
  if (currentPreference !== AUTO_LOCALE) return;
  const locale = detectLocale();
  if (locale === currentLocale) return;
  currentLocale = locale;
  syncDocumentLang(locale);
  notifySubscribers();
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

export function subscribeLocale(listener: () => void): () => void {
  const shouldStartListeners = subscribers.size === 0;
  subscribers.add(listener);
  if (shouldStartListeners) startListeners();

  return () => {
    subscribers.delete(listener);
    if (subscribers.size === 0) stopListeners();
  };
}

export function initializeLocale(): Locale {
  const preference = readStoredPreference();
  currentPreference = preference;
  currentLocale = resolvePreference(preference);
  syncDocumentLang(currentLocale);
  notifySubscribers();
  return currentLocale;
}

export function getLocale(): Locale {
  return currentLocale;
}

export function getLocalePreference(): LocalePreference {
  return currentPreference;
}

export function setLocale(preference: LocalePreference): void {
  const requestedPreference = normalizePreference(preference);
  writeStoredPreference(requestedPreference);

  currentPreference = requestedPreference;
  currentLocale = resolvePreference(requestedPreference);
  syncDocumentLang(currentLocale);
  notifySubscribers();
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
