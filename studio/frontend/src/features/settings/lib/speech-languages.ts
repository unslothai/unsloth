// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface SpeechLanguageOption {
  value: string;
  label: string;
  ttsName?: string;
}

// Shared picker options for web and desktop. Support depends on the selected speech engine.
export const SPEECH_LANGUAGES: SpeechLanguageOption[] = [
  { value: "auto", label: "" },
  { value: "en-US", label: "English (US)", ttsName: "English" },
  { value: "en-GB", label: "English (UK)", ttsName: "English" },
  { value: "zh-CN", label: "中文 (简体)", ttsName: "Chinese" },
  { value: "ja-JP", label: "日本語", ttsName: "Japanese" },
  { value: "ko-KR", label: "한국어", ttsName: "Korean" },
  { value: "es-ES", label: "Español", ttsName: "Spanish" },
  { value: "fr-FR", label: "Français", ttsName: "French" },
  { value: "de-DE", label: "Deutsch", ttsName: "German" },
  { value: "it-IT", label: "Italiano", ttsName: "Italian" },
  {
    value: "pt-BR",
    label: "Português (Brasil)",
    ttsName: "Portuguese",
  },
  { value: "ru-RU", label: "Русский", ttsName: "Russian" },
  { value: "hi-IN", label: "हिन्दी", ttsName: "Hindi" },
  { value: "ar-SA", label: "العربية", ttsName: "Arabic" },
  { value: "el-GR", label: "Ελληνικά", ttsName: "Greek" },
];

export function resolveTtsLanguageName(language: string): string | undefined {
  const normalized = language.trim().replaceAll("_", "-").toLowerCase();
  if (!normalized || normalized === "auto") {
    return undefined;
  }
  return (
    SPEECH_LANGUAGES.find((option) => option.value.toLowerCase() === normalized)
      ?.ttsName ?? language.trim()
  );
}

export function ttsPreviewText(language: string): string {
  return resolveTtsLanguageName(language) === "Greek"
    ? "Γεια σας από το Unsloth! Αυτή είναι μια προεπισκόπηση της επιλεγμένης φωνής."
    : "Hello from Unsloth! This is a preview of the selected voice.";
}
