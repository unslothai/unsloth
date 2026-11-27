// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TranslateLanguage = {
  code: string;
  name: string;
  /** BCP 47 tag for `Intl.DisplayNames`, when it differs from `code`. */
  intl?: string;
  /** Not one of Hy-MT2's languages: offered only with a chat model. */
  chatOnly?: true;
};

/** The languages Hy-MT2 was trained on (its model card), most used first, then ones only chat models translate. */
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
  { code: "hmn", name: "Hmong", chatOnly: true },
  { code: "am", name: "Amharic", chatOnly: true },
  { code: "sw", name: "Swahili", chatOnly: true },
  { code: "ha", name: "Hausa", chatOnly: true },
  { code: "yo", name: "Yoruba", chatOnly: true },
  { code: "ig", name: "Igbo", chatOnly: true },
  { code: "zu", name: "Zulu", chatOnly: true },
  { code: "af", name: "Afrikaans", chatOnly: true },
  { code: "so", name: "Somali", chatOnly: true },
];

export const DEFAULT_TARGET_LANGUAGE = "en";

export function findTranslateLanguage(
  code: string,
): TranslateLanguage | undefined {
  return TRANSLATE_LANGUAGES.find((language) => language.code === code);
}

/** What a model translates: Hy-MT2 (`dedicated`) only its own languages. */
export function translateLanguagesFor(
  dedicated: boolean,
): readonly TranslateLanguage[] {
  return dedicated
    ? TRANSLATE_LANGUAGES.filter((language) => !language.chatOnly)
    : TRANSLATE_LANGUAGES;
}

export function translatesLanguage(code: string, dedicated: boolean): boolean {
  const language = findTranslateLanguage(code);
  return language !== undefined && !(dedicated && language.chatOnly);
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

const SPEECH_REGIONS: Record<string, string> = {
  en: "en-US",
  fr: "fr-FR",
  es: "es-ES",
  pt: "pt-BR",
  de: "de-DE",
  nl: "nl-NL",
  it: "it-IT",
  ar: "ar-SA",
};

export function translateSpeechTag(
  code: string,
  preferred: readonly string[] = typeof navigator === "undefined"
    ? []
    : (navigator.languages ?? []),
): string {
  const fixed = SPEECH_TAGS[code];
  if (fixed) {
    return fixed;
  }
  const own = preferred.find(
    (tag) => tag.includes("-") && tag.split("-")[0].toLowerCase() === code,
  );
  return own ?? SPEECH_REGIONS[code] ?? code;
}

export function suggestedTargetLanguage(preferred: readonly string[]): string {
  for (const tag of preferred) {
    const lower = tag.toLowerCase();
    if (/^zh-(tw|hk|mo)\b|hant/.test(lower)) return "zh-Hant";
    const primary = lower.split("-")[0];
    const match = translateLanguagesFor(true).find(
      (language) => language.code === primary || language.intl === primary,
    );
    if (match) return match.code;
  }
  return DEFAULT_TARGET_LANGUAGE;
}
