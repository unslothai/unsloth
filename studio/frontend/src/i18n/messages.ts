// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getLocale } from "./locale-store";
import { en } from "./locales/en";
import { zhCN } from "./locales/zh-CN";
import { ptBR } from "./locales/pt-br";
import { ja } from "./locales/ja";
import { es } from "./locales/es";
import { hi } from "./locales/hi";
import { ar } from "./locales/ar";
import { fr } from "./locales/fr";
import { ru } from "./locales/ru";
import { de } from "./locales/de";
import { ko } from "./locales/ko";
import type { InterpolationValues, MessageKey } from "./types";

// dir sets documentElement.dir; Arabic stays ltr (CSS still physical-direction) but renders rtl via bidi.
export const LOCALES = {
  en: { label: "English", nativeLabel: "English", dir: "ltr" },
  "zh-CN": { label: "Chinese (Simplified)", nativeLabel: "简体中文", dir: "ltr" },
  ja: { label: "Japanese", nativeLabel: "日本語", dir: "ltr" },
  ko: { label: "Korean", nativeLabel: "한국어", dir: "ltr" },
  es: { label: "Spanish", nativeLabel: "Español", dir: "ltr" },
  "pt-BR": { label: "Portuguese (Brazil)", nativeLabel: "Português (Brasil)", dir: "ltr" },
  fr: { label: "French", nativeLabel: "Français", dir: "ltr" },
  de: { label: "German", nativeLabel: "Deutsch", dir: "ltr" },
  ru: { label: "Russian", nativeLabel: "Русский", dir: "ltr" },
  hi: { label: "Hindi", nativeLabel: "हिन्दी", dir: "ltr" },
  ar: { label: "Arabic", nativeLabel: "العربية", dir: "ltr" },
} as const;

export type Locale = keyof typeof LOCALES;
export type TranslationKey = MessageKey<typeof en>;

export const messages = {
  en,
  "zh-CN": zhCN,
  ja,
  ko,
  es,
  "pt-BR": ptBR,
  fr,
  de,
  ru,
  hi,
  ar,
} as const;

const PLACEHOLDER_PATTERN = /\{([a-zA-Z0-9_]+)\}/g;

function readMessage(tree: unknown, key: string): string | undefined {
  let cursor = tree;
  for (const segment of key.split(".")) {
    if (
      cursor === null ||
      typeof cursor !== "object" ||
      !Object.prototype.hasOwnProperty.call(cursor, segment)
    ) {
      return undefined;
    }
    cursor = (cursor as Record<string, unknown>)[segment];
  }
  return typeof cursor === "string" ? cursor : undefined;
}

function interpolate(
  template: string,
  values: InterpolationValues | undefined,
): string {
  if (!values) return template;

  return template.replace(PLACEHOLDER_PATTERN, (match, name: string) => {
    if (!Object.prototype.hasOwnProperty.call(values, name)) return match;
    const value = values[name];
    return value === null || value === undefined ? "" : String(value);
  });
}

function warnMissingEnglishMessage(key: string): void {
  // Optional chain so translate() also works outside Vite (Node tooling).
  if (import.meta.env?.DEV) {
    console.warn(`[i18n] Missing English translation for key "${key}".`);
  }
}

export function translate(
  key: TranslationKey,
  values?: InterpolationValues,
  locale: Locale = getLocale(),
): string {
  const localized = readMessage(messages[locale], key);
  const fallback = localized ?? readMessage(messages.en, key);

  if (fallback === undefined) {
    warnMissingEnglishMessage(key);
    return key;
  }

  return interpolate(fallback, values);
}

export function isSupportedLocale(value: unknown): value is Locale {
  return (
    typeof value === "string" &&
    Object.prototype.hasOwnProperty.call(LOCALES, value)
  );
}