// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// free of app imports so the node test runner can load it directly.

/** Target languages offered by the picker; the prompt names them in English, which every model reads. */
export const TRANSLATE_LANGUAGES = [
  { code: "en", name: "English" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "es", name: "Spanish" },
  { code: "pt", name: "Portuguese" },
  { code: "it", name: "Italian" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "ru", name: "Russian" },
  { code: "zh", name: "Chinese" },
  { code: "ar", name: "Arabic" },
  { code: "fa", name: "Persian" },
  { code: "id", name: "Indonesian" },
  { code: "ms", name: "Malay" },
  { code: "ne", name: "Nepali" },
  { code: "pl", name: "Polish" },
  { code: "ro", name: "Romanian" },
  { code: "sr", name: "Serbian" },
  { code: "sv", name: "Swedish" },
  { code: "tr", name: "Turkish" },
  { code: "uk", name: "Ukrainian" },
  { code: "vi", name: "Vietnamese" },
  { code: "hi", name: "Hindi" },
  { code: "bn", name: "Bengali" },
] as const;

export type TranslateLanguageCode =
  (typeof TRANSLATE_LANGUAGES)[number]["code"];
/** "auto" lets the model detect the source language. */
export type TranslateSourceLanguage = TranslateLanguageCode | "auto";

export const TRANSLATE_MODES = ["automatic", "literal", "semantic"] as const;
export type TranslateMode = (typeof TRANSLATE_MODES)[number];

export const TRANSLATE_TONES = ["balanced", "formal", "casual"] as const;
export type TranslateTone = (typeof TRANSLATE_TONES)[number];

export type TranslateOptions = {
  source: TranslateSourceLanguage;
  target: TranslateLanguageCode;
  mode: TranslateMode;
  tone: TranslateTone;
};

export type TranslationEntry = {
  id: string;
  createdAt: number;
  source: TranslateSourceLanguage;
  target: TranslateLanguageCode;
  mode: TranslateMode;
  tone: TranslateTone;
  input: string;
  output: string;
  model: string;
};

const LANGUAGE_NAMES: ReadonlyMap<string, string> = new Map(
  TRANSLATE_LANGUAGES.map((language) => [language.code, language.name]),
);

export function isTranslateLanguage(
  value: unknown,
): value is TranslateLanguageCode {
  return typeof value === "string" && LANGUAGE_NAMES.has(value);
}

export function isTranslateSource(
  value: unknown,
): value is TranslateSourceLanguage {
  return value === "auto" || isTranslateLanguage(value);
}

export function isTranslateMode(value: unknown): value is TranslateMode {
  return (
    typeof value === "string" &&
    (TRANSLATE_MODES as readonly string[]).includes(value)
  );
}

export function isTranslateTone(value: unknown): value is TranslateTone {
  return (
    typeof value === "string" &&
    (TRANSLATE_TONES as readonly string[]).includes(value)
  );
}

export function languageName(code: TranslateSourceLanguage): string {
  return code === "auto" ? "Auto-detected" : (LANGUAGE_NAMES.get(code) ?? code);
}

const MODE_RULES: Record<TranslateMode, string> = {
  automatic:
    "Choose between a literal and a meaning-based translation, whichever reads most naturally for this text, and match the tone of the original.",
  literal:
    "Translate word for word. Keep the original sentence structure and map each phrase directly, even where the result reads less naturally.",
  semantic:
    "Translate idea for idea. Capture the meaning, nuance and natural flow of the original in idiomatic target-language phrasing.",
};

const TONE_RULES: Record<TranslateTone, string> = {
  balanced: "Use a neutral, balanced register.",
  formal: "Use a formal, polite register.",
  casual:
    "Use a casual, conversational register. Everyday slang is allowed when it fits.",
};

/** The translator's instructions, also shown by the "Show system prompt" toggle. */
export function buildTranslateSystemPrompt(options: TranslateOptions): string {
  const target = languageName(options.target);
  const source =
    options.source === "auto"
      ? "Detect the language of the text."
      : `The text is in ${languageName(options.source)}.`;
  const rules = [
    "Reply with the translation only: no preamble, no quotes around it and no explanation.",
    "Keep line breaks, lists, Markdown, code, URLs, numbers and names as they are.",
    "Never invent or fabricate translations, idioms or cultural references.",
    "Never sacrifice accuracy for brevity or style.",
    options.tone === "casual"
      ? null
      : "Do not use slang, emojis or overly casual language.",
    `If the text is already in ${target}, return it unchanged.`,
    'If a passage is ambiguous, translate its most likely reading, then add one final line starting with "Note:" that names the ambiguity.',
  ].filter((rule): rule is string => rule !== null);
  return [
    `You are a translator. Translate the user's text into ${target}.`,
    source,
    MODE_RULES[options.mode],
    TONE_RULES[options.tone],
    "",
    "Rules:",
    ...rules.map((rule) => `- ${rule}`),
  ].join("\n");
}

/** Drops reasoning (closed, still streaming, or opened by the chat template) and outer whitespace. */
export function cleanTranslation(raw: string): string {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/^[\s\S]*<\/think>/i, "")
    .replace(/<think>[\s\S]*$/i, "")
    .trim();
}

function entryLabel(entry: TranslationEntry): string {
  return `${languageName(entry.source)} -> ${languageName(entry.target)}`;
}

/** The text a "Chat with translated text" handoff puts in the composer: the recent pairs, oldest first. */
export function buildChatHandoff(entries: readonly TranslationEntry[]): string {
  const blocks = [...entries]
    .sort((a, b) => a.createdAt - b.createdAt)
    .map(
      (entry) =>
        `Original (${languageName(entry.source)}):\n${entry.input}\n\nTranslation (${languageName(entry.target)}):\n${entry.output}`,
    );
  if (blocks.length === 0) return "";
  return `Here are my recent translations:\n\n${blocks.join("\n\n---\n\n")}\n\n`;
}

/** Plain text with a timestamp per entry, oldest first, for UnslothTranslateFullHistory.txt. */
export function exportTranslateHistory(
  entries: readonly TranslationEntry[],
): string {
  return [...entries]
    .sort((a, b) => a.createdAt - b.createdAt)
    .map((entry) =>
      [
        `[${new Date(entry.createdAt).toISOString()}] ${entryLabel(entry)} (${entry.mode}, ${entry.tone}, ${entry.model})`,
        entry.input,
        "",
        entry.output,
      ].join("\n"),
    )
    .join("\n\n---\n\n")
    .concat(entries.length > 0 ? "\n" : "");
}

export const TRANSLATE_HISTORY_FILE_NAME = "UnslothTranslateFullHistory.txt";
