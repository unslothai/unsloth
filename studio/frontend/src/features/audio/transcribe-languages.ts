// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Transcribe's language list. Speech-to-text runtimes take ISO codes; the TTS LanguageSelect is
// keyed by names, so it is not reused here. Empty means the model detects the language.

export const TRANSCRIBE_LANGUAGES: readonly { code: string; name: string }[] = [
  { code: "", name: "Detect automatically" },
  { code: "en", name: "English" },
  { code: "zh", name: "Chinese" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
  { code: "nl", name: "Dutch" },
  { code: "pl", name: "Polish" },
  { code: "tr", name: "Turkish" },
  { code: "uk", name: "Ukrainian" },
  { code: "vi", name: "Vietnamese" },
  { code: "id", name: "Indonesian" },
  { code: "th", name: "Thai" },
  { code: "sv", name: "Swedish" },
  { code: "da", name: "Danish" },
  { code: "fi", name: "Finnish" },
  { code: "no", name: "Norwegian" },
  { code: "cs", name: "Czech" },
  { code: "el", name: "Greek" },
  { code: "he", name: "Hebrew" },
  { code: "hu", name: "Hungarian" },
  { code: "ro", name: "Romanian" },
  { code: "ms", name: "Malay" },
  { code: "yue", name: "Cantonese" },
];

/** The list a model can use: an English-only model gets no choice, a short list keeps Auto. */
export function transcribeLanguagesFor(
  modelLanguages: readonly string[] | null | undefined,
): readonly { code: string; name: string }[] {
  if (!modelLanguages || modelLanguages.length === 0)
    return TRANSCRIBE_LANGUAGES;
  return TRANSCRIBE_LANGUAGES.filter(
    (language) =>
      language.code === "" || modelLanguages.includes(language.code),
  );
}
