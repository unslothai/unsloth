// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getLocale } from "./locale-store";
import { en } from "./locales/en";
import type { InterpolationValues, MessageKey, MessageTree } from "./types";

export const LOCALES = {
  en: { label: "English", nativeLabel: "English" },
  "zh-CN": { label: "Chinese (Simplified)", nativeLabel: "简体中文" },
  ja: { label: "Japanese", nativeLabel: "日本語" },
  ko: { label: "Korean", nativeLabel: "한국어" },
  es: { label: "Spanish", nativeLabel: "Español" },
  "pt-BR": { label: "Portuguese (Brazil)", nativeLabel: "Português (Brasil)" },
  fr: { label: "French", nativeLabel: "Français" },
  de: { label: "German", nativeLabel: "Deutsch" },
  it: { label: "Italian", nativeLabel: "Italiano" },
  ru: { label: "Russian", nativeLabel: "Русский" },
  hi: { label: "Hindi", nativeLabel: "हिन्दी" },
  ar: { label: "Arabic", nativeLabel: "العربية" },
} as const;

export type Locale = keyof typeof LOCALES;
export type TranslationKey = MessageKey<typeof en>;

const loadedMessages: Partial<Record<Locale, MessageTree>> = {
  en,
};
export const messages = loadedMessages as typeof loadedMessages & {
  en: typeof en;
};

const localeLoaders: Record<
  Exclude<Locale, "en">,
  () => Promise<MessageTree>
> = {
  "zh-CN": () => import("./locales/zh-CN").then((module) => module.zhCN),
  ja: () => import("./locales/ja").then((module) => module.ja),
  ko: () => import("./locales/ko").then((module) => module.ko),
  es: () => import("./locales/es").then((module) => module.es),
  "pt-BR": () => import("./locales/pt-br").then((module) => module.ptBR),
  fr: () => import("./locales/fr").then((module) => module.fr),
  de: () => import("./locales/de").then((module) => module.de),
  it: () => import("./locales/it").then((module) => module.it),
  ru: () => import("./locales/ru").then((module) => module.ru),
  hi: () => import("./locales/hi").then((module) => module.hi),
  ar: () => import("./locales/ar").then((module) => module.ar),
};

const localeLoads = new Map<Locale, Promise<void>>();

export function loadLocaleMessages(locale: Locale): Promise<void> | undefined {
  if (loadedMessages[locale] !== undefined) return undefined;
  const pending = localeLoads.get(locale);
  if (pending) return pending;

  if (locale === "en") return undefined;
  const load = localeLoaders[locale]()
    .then((loaded) => {
      loadedMessages[locale] = loaded;
    })
    .finally(() => {
      localeLoads.delete(locale);
    });
  localeLoads.set(locale, load);
  return load;
}

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
