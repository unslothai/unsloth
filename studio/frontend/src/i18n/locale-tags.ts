// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No imports: the build's locale preload (vite-plugin-locale-preload.ts) reads this table and
// ships matchLocaleTag's source in its boot script, so both pick the same catalog.

export const LOCALES = {
  en: { label: "English", nativeLabel: "English" },
  "zh-CN": { label: "Chinese (Simplified)", nativeLabel: "简体中文" },
  "zh-TW": {
    label: "Chinese (Traditional, Taiwan)",
    nativeLabel: "繁體中文（台灣）",
  },
  ja: { label: "Japanese", nativeLabel: "日本語" },
  ko: { label: "Korean", nativeLabel: "한국어" },
  es: { label: "Spanish (Spain)", nativeLabel: "Español (España)" },
  "es-419": {
    label: "Spanish (Latin America)",
    nativeLabel: "Español (Latinoamérica)",
  },
  "pt-BR": { label: "Portuguese (Brazil)", nativeLabel: "Português (Brasil)" },
  fr: { label: "French", nativeLabel: "Français" },
  de: { label: "German", nativeLabel: "Deutsch" },
  it: { label: "Italian", nativeLabel: "Italiano" },
  ru: { label: "Russian", nativeLabel: "Русский" },
  sv: { label: "Swedish", nativeLabel: "Svenska" },
  hi: { label: "Hindi", nativeLabel: "हिन्दी" },
  ar: { label: "Arabic", nativeLabel: "العربية" },
  he: { label: "Hebrew", nativeLabel: "עברית" },
  id: { label: "Indonesian", nativeLabel: "Bahasa Indonesia" },
  vi: { label: "Vietnamese", nativeLabel: "Tiếng Việt" },
  tr: { label: "Turkish", nativeLabel: "Türkçe" },
  pl: { label: "Polish", nativeLabel: "Polski" },
  uk: { label: "Ukrainian", nativeLabel: "Українська" },
  th: { label: "Thai", nativeLabel: "ไทย" },
  fa: { label: "Persian", nativeLabel: "فارسی" },
  cs: { label: "Czech", nativeLabel: "Čeština" },
  nl: { label: "Dutch", nativeLabel: "Nederlands" },
} as const;

/** The supported locale for one browser language tag, or null. Must not reference anything
 * outside its own body: the boot script runs a copy of its source. */
export function matchLocaleTag(
  tag: string,
  locales: readonly string[],
): string | null {
  const lower = tag.toLowerCase();
  const exact = locales.find((locale) => locale.toLowerCase() === lower);
  if (exact) return exact;
  const parts = lower.split("-");
  const language = parts[0];
  if (language === "zh") {
    const traditional = lower.includes("hant")
      ? true
      : lower.includes("hans")
        ? false
        : parts.includes("tw") || parts.includes("hk") || parts.includes("mo");
    return traditional ? "zh-TW" : "zh-CN";
  }
  if (language === "es") {
    const latinAmerica =
      "419 mx ar co cl pe ve ec gt cu bo do hn py sv ni cr pr pa uy us".split(
        " ",
      );
    const region = parts
      .slice(1)
      .find((part) => /^([a-z]{2}|\d{3})$/.test(part));
    return region && latinAmerica.includes(region) ? "es-419" : "es";
  }
  return (
    locales.find((locale) => locale.toLowerCase().split("-")[0] === language) ??
    null
  );
}
