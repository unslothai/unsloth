// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getLocale } from "./locale-store";
import { en } from "./locales/en";
import type { InterpolationValues, MessageKey, MessageTree } from "./types";

export const LOCALES = {
  en: { label: "English", nativeLabel: "English" },
  "zh-CN": { label: "Chinese (Simplified)", nativeLabel: "简体中文" },
  he: { label: "Hebrew", nativeLabel: "עברית" },
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

type LazyLocale = Exclude<Locale, "en">;

const localeLoaders: Record<LazyLocale, () => Promise<unknown>> = {
  "zh-CN": () => import("./locales/zh-CN"),
  he: () => import("./locales/he"),
  ja: () => import("./locales/ja"),
  ko: () => import("./locales/ko"),
  es: () => import("./locales/es"),
  "pt-BR": () => import("./locales/pt-br"),
  fr: () => import("./locales/fr"),
  de: () => import("./locales/de"),
  it: () => import("./locales/it"),
  ru: () => import("./locales/ru"),
  hi: () => import("./locales/hi"),
  ar: () => import("./locales/ar"),
};

/** A catalog exports its own tag with the separator dropped: zh-CN -> zhCN. */
function readCatalog(module: unknown, locale: LazyLocale): MessageTree {
  const name = locale.replace("-", "");
  const catalog = (module as Record<string, unknown> | null)?.[name];
  if (catalog === undefined) {
    throw new Error(`Locale catalog "${locale}" has no "${name}" export.`);
  }
  return catalog as MessageTree;
}

export const CATALOG_RETRY_PARAM = "catalogRetry";

const CHUNK_URL_PATTERN = /\bhttps?:\/\/[^\s"'`)]+/;

let catalogRetryCount = 0;

/**
 * The URL to re-request a failed catalog from, or null when none can be read.
 *
 * Chrome, Edge and Firefox before 155 keep a failed module in the module map
 * keyed by URL, so re-running the same import resolves to the stored failure
 * without touching the network: dropping our own promise is not enough to make
 * a retry a retry. A one-off query gives the request its own module map key and
 * its own HTTP cache key, while the hashed filename it is appended to is
 * unchanged, so the first load of every catalog keeps its normal long-lived
 * caching and nothing about it moves until something has actually failed.
 *
 * The URL comes out of the browser's own message ("Failed to fetch dynamically
 * imported module: <url>"), which is the only place the built chunk's URL is
 * exposed to us; Safari reports no URL there, and it is also the one engine
 * that already re-requests on its own.
 */
export function catalogRetryUrl(
  error: unknown,
  previousUrl: string | null = null,
): string | null {
  const message = error instanceof Error ? error.message : String(error ?? "");
  const found = message.match(CHUNK_URL_PATTERN)?.[0] ?? previousUrl;
  if (found === null) return null;
  let url: URL;
  try {
    url = new URL(found);
  } catch {
    return null;
  }
  // A preload failure names the stylesheet, not the module that wanted it.
  if (url.pathname.endsWith(".css")) return null;
  catalogRetryCount += 1;
  url.searchParams.set(CATALOG_RETRY_PARAM, String(catalogRetryCount));
  return url.href;
}

export type CatalogImporter = (
  locale: LazyLocale,
  retryUrl: string | null,
) => Promise<unknown>;

function importCatalog(
  locale: LazyLocale,
  retryUrl: string | null,
): Promise<unknown> {
  if (retryUrl === null) return localeLoaders[locale]();
  return import(/* @vite-ignore */ retryUrl);
}

const localeLoads = new Map<Locale, Promise<void>>();
const catalogRetryUrls = new Map<Locale, string>();

export function loadLocaleMessages(
  locale: Locale,
  importer: CatalogImporter = importCatalog,
): Promise<void> | undefined {
  if (loadedMessages[locale] !== undefined) return undefined;
  const pending = localeLoads.get(locale);
  if (pending) return pending;

  if (locale === "en") return undefined;
  const retryUrl = catalogRetryUrls.get(locale) ?? null;
  const load = importer(locale, retryUrl)
    .then(
      (module) => {
        loadedMessages[locale] = readCatalog(module, locale);
        catalogRetryUrls.delete(locale);
      },
      (error: unknown) => {
        const nextUrl = catalogRetryUrl(error, retryUrl);
        if (nextUrl === null) catalogRetryUrls.delete(locale);
        else catalogRetryUrls.set(locale, nextUrl);
        throw error;
      },
    )
    .finally(() => {
      // Only if it is still ours: a load evicted by its caller's timeout may
      // already have been replaced by the retry it made room for.
      if (localeLoads.get(locale) === load) localeLoads.delete(locale);
    });
  localeLoads.set(locale, load);
  return load;
}

/**
 * Stop deduplicating onto a catalog load, so the next request starts its own.
 *
 * A load that never settles never reaches the `.finally()` above, so its entry
 * would hand every later pick of that language the same permanently pending
 * promise and the choice could not be retried without reloading the app.
 * Whoever bounded the wait evicts the load it gave up on; the promise itself is
 * left running, so a late arrival still populates the catalog.
 *
 * `load` identifies the attempt being forgotten, so a newer one for the same
 * locale is never evicted by an older caller's timeout.
 */
export function forgetLocaleLoad(locale: Locale, load?: Promise<void>): void {
  if (load !== undefined && localeLoads.get(locale) !== load) return;
  localeLoads.delete(locale);
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
