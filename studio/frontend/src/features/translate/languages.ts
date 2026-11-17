// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TranslateLanguage = {
  code: string;
  name: string;
  /** BCP 47 tag for `Intl.DisplayNames`, when it differs from `code`. */
  intl?: string;
};

/** The languages Hy-MT2 was trained on (its model card), most used first. */
export const TRANSLATE_LANGUAGES: readonly TranslateLanguage[] = [
  { code: "en", name: "English" },
  { code: "zh", name: "Chinese", intl: "zh-Hans" },
  { code: "zh-Hant", name: "Traditional Chinese" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "pt", name: "Portuguese" },
  { code: "it", name: "Italian" },
  { code: "ru", name: "Russian" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
  { code: "tr", name: "Turkish" },
  { code: "nl", name: "Dutch" },
  { code: "pl", name: "Polish" },
  { code: "uk", name: "Ukrainian" },
  { code: "cs", name: "Czech" },
  { code: "vi", name: "Vietnamese" },
  { code: "th", name: "Thai" },
  { code: "id", name: "Indonesian" },
  { code: "ms", name: "Malay" },
  { code: "tl", name: "Filipino", intl: "fil" },
  { code: "he", name: "Hebrew" },
  { code: "fa", name: "Persian" },
  { code: "ur", name: "Urdu" },
  { code: "bn", name: "Bengali" },
  { code: "ta", name: "Tamil" },
  { code: "te", name: "Telugu" },
  { code: "mr", name: "Marathi" },
  { code: "gu", name: "Gujarati" },
  { code: "km", name: "Khmer" },
  { code: "my", name: "Burmese" },
  { code: "yue", name: "Cantonese" },
  { code: "bo", name: "Tibetan" },
  { code: "kk", name: "Kazakh" },
  { code: "mn", name: "Mongolian" },
  { code: "ug", name: "Uyghur" },
];

export const DEFAULT_TARGET_LANGUAGE = "en";

/**
 * Keep the picker easy to scan without changing the model's preferred-language
 * order used when suggesting a target from the browser locale.
 */
export function sortTranslateLanguages(
  languages: readonly TranslateLanguage[],
  locale: string,
): TranslateLanguage[] {
  const labels = new Map(
    languages.map((language) => [
      language.code,
      translateLanguageLabel(language, locale),
    ]),
  );
  const collator = new Intl.Collator(locale, { sensitivity: "base" });
  return [...languages].sort(
    (a, b) =>
      collator.compare(
        labels.get(a.code) ?? a.name,
        labels.get(b.code) ?? b.name,
      ) || a.code.localeCompare(b.code),
  );
}

export function findTranslateLanguage(
  code: string,
): TranslateLanguage | undefined {
  return TRANSLATE_LANGUAGES.find((language) => language.code === code);
}

export function translateLanguageLabel(
  language: TranslateLanguage,
  locale: string,
): string {
  try {
    const label = new Intl.DisplayNames([locale], { type: "language" }).of(
      language.intl ?? language.code,
    );
    // An unknown tag comes back unchanged.
    if (label && label !== (language.intl ?? language.code)) return label;
  } catch {
    // Older engines throw on a tag they cannot parse.
  }
  return language.name;
}

const SPEECH_TAGS: Record<string, string> = {
  zh: "zh-CN",
  "zh-Hant": "zh-TW",
  yue: "zh-HK",
  tl: "fil",
};

export function translateSpeechTag(code: string): string {
  return SPEECH_TAGS[code] ?? code;
}

export function suggestedTargetLanguage(preferred: readonly string[]): string {
  for (const tag of preferred) {
    const lower = tag.toLowerCase();
    if (/^zh-(tw|hk|mo)\b|hant/.test(lower)) return "zh-Hant";
    const primary = lower.split("-")[0];
    const match = TRANSLATE_LANGUAGES.find(
      (language) => language.code === primary || language.intl === primary,
    );
    if (match) return match.code;
  }
  return DEFAULT_TARGET_LANGUAGE;
}
