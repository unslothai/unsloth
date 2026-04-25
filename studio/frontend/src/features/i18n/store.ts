// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  LOCALE_OPTIONS,
  MESSAGES,
  type LocaleCode,
  type TranslationKey,
} from "./messages";

const DEFAULT_LOCALE: LocaleCode = "en";
const LOCALE_STORAGE_KEY = "unsloth_locale";

const SUPPORTED_LOCALES = new Set<LocaleCode>(
  LOCALE_OPTIONS.map((option) => option.code),
);

function normalizeLocale(input: string | null | undefined): LocaleCode | null {
  if (!input) return null;
  if (SUPPORTED_LOCALES.has(input as LocaleCode)) {
    return input as LocaleCode;
  }

  const lowered = input.trim().toLowerCase();
  if (!lowered) return null;

  if (lowered.startsWith("zh")) return "zh-CN";
  if (lowered.startsWith("en")) return "en";

  return null;
}

function loadInitialLocale(): LocaleCode {
  if (typeof window === "undefined") return DEFAULT_LOCALE;

  try {
    const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    const normalizedStored = normalizeLocale(stored);
    if (normalizedStored) return normalizedStored;
  } catch {
    // ignore storage failures
  }

  for (const locale of navigator.languages ?? []) {
    const normalized = normalizeLocale(locale);
    if (normalized) return normalized;
  }

  return normalizeLocale(navigator.language) ?? DEFAULT_LOCALE;
}

function saveLocale(locale: LocaleCode): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  } catch {
    // ignore storage failures
  }
}

interface I18nState {
  locale: LocaleCode;
  setLocale: (locale: LocaleCode) => void;
}

export const useI18nStore = create<I18nState>((set) => ({
  locale: loadInitialLocale(),
  setLocale: (locale) => {
    saveLocale(locale);
    set({ locale });
  },
}));

export function translate(locale: LocaleCode, key: TranslationKey): string {
  return MESSAGES[locale]?.[key] ?? MESSAGES.en[key];
}
