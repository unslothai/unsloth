// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// ISO codes, unlike the name-keyed TTS LanguageSelect; "" means the model detects it.

const NAMES = new Intl.DisplayNames(["en"], { type: "language" });

export const TRANSCRIBE_LANGUAGES: readonly { code: string; name: string }[] = [
  { code: "", name: "Detect automatically" },
  ...(
    "en zh es fr de it pt ru ja ko ar hi nl pl tr uk vi id th sv da fi no cs " +
    "el he hu ro ms yue"
  )
    .split(" ")
    .map((code) => ({ code, name: NAMES.of(code) ?? code })),
];

/** The saved language if the model lists it, else "" (detect), never one hidden behind Auto. */
export function transcribeLanguageFor(
  saved: string,
  languages: readonly { code: string }[],
): string {
  return saved && languages.some((entry) => entry.code === saved) ? saved : "";
}

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
