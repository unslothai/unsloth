// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Transcript cleanup for recorded dictation.
//
// The model returns the sounds it heard, not the text the speaker meant: "um"
// and "uh" are speech, repeated words are stutters, and a spoken "comma" is a
// word until it becomes punctuation. Everything here is deterministic and runs
// per segment, so the composer sees clean text as each segment finishes and the
// system-wide controller can insert the same text without a second pass.

/** Filler tokens that are not lexical words in any language the dictation
 *  models can output. Anything that is a real word somewhere ("um" in
 *  Portuguese, "ah" in Spanish) is language-gated below. */
const UNIVERSAL_FILLER_WORDS = [
  "uh",
  "uhm",
  "umm",
  "uhh",
  "uhhh",
  "ehh",
  "ehm",
  "ahm",
  "hmm",
  "hm",
  "mmm",
];

/** Fillers that are only safe to remove when the output language is known. */
function gatedFillerWords(language: string): string[] {
  const base = language.trim().toLowerCase().split(/[-_]/, 1)[0] ?? "";
  switch (base) {
    case "en":
      return ["um", "ah", "eh"];
    case "de":
      return ["äh", "ähm"];
    case "fr":
      return ["euh"];
    default:
      return [];
  }
}

function normalizeWhitespace(text: string): string {
  const normalized = text
    .replace(/[ \t]{2,}/g, " ")
    .replace(/[ \t]*\n[ \t]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n");
  const core = normalized.trim();
  // Spoken "new line" and "new paragraph" are whitespace, so trimming would
  // delete them. Keep the break at either edge and let the join decide whether
  // it has anything to separate.
  if (!core) return normalized.includes("\n") ? normalized.replace(/[ \t]/g, "") : "";
  const leading = /^\n+/.exec(normalized)?.[0] ?? "";
  const trailing = /\n+$/.exec(normalized)?.[0] ?? "";
  return `${leading}${core}${trailing}`;
}

function opensSentence(text: string): boolean {
  const trimmed = text.trimEnd();
  return (
    trimmed.length === 0 ||
    /[.!?\u2026]\s*$/.test(trimmed) ||
    /\n\s*$/.test(trimmed)
  );
}

/** Uppercase the first letter or digit in `text`, leaving leading punctuation alone. */
function capitalizeFirst(text: string): string {
  const match = /[\p{L}\p{N}]/u.exec(text);
  if (!match) return text;
  const index = match.index;
  return `${text.slice(0, index)}${text[index].toUpperCase()}${text.slice(index + 1)}`;
}

/** Remove filler words while handing a sentence-opening capital to the word
 *  that takes the filler's place: "Um, so I think" -> "So I think". */
export function removeFillerWords(
  text: string,
  language: string,
  enabled = true,
): string {
  if (!enabled || !text) return text;
  const fillers = new Set(
    [...UNIVERSAL_FILLER_WORDS, ...gatedFillerWords(language)].map((word) =>
      word.toLowerCase(),
    ),
  );
  if (fillers.size === 0) return text;

  const parts = text.split(/(\s+)/);
  let kept = "";
  let capitalOwed = false;
  let previousWasFiller = false;

  for (const part of parts) {
    if (!part) continue;
    if (/^\s+$/.test(part)) {
      kept += part;
      continue;
    }
    const core = part.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
    if (core && fillers.has(core.toLowerCase())) {
      if (/^\p{Lu}/u.test(core) && opensSentence(kept)) capitalOwed = true;
      previousWasFiller = true;
      continue;
    }
    // A comma or full stop left behind as its own token belongs to the filler.
    if (previousWasFiller && /^[,.;:!?]+$/.test(part)) {
      previousWasFiller = false;
      continue;
    }
    previousWasFiller = false;
    if (capitalOwed && /[\p{L}\p{N}]/u.test(part)) {
      capitalOwed = false;
      kept += capitalizeFirst(part);
    } else {
      kept += part;
    }
  }

  return normalizeWhitespace(kept);
}

function collapseStutterLine(line: string): string {
  const words = line.split(/[ \t]+/).filter(Boolean);
  if (words.length === 0) return "";
  const kept: string[] = [];
  let index = 0;
  while (index < words.length) {
    const word = words[index];
    const lower = word.toLowerCase();
    if (/^\p{L}+$/u.test(lower)) {
      let count = 1;
      while (
        index + count < words.length &&
        words[index + count].toLowerCase() === lower
      ) {
        count += 1;
      }
      kept.push(word);
      index += count >= 3 ? count : 1;
    } else {
      kept.push(word);
      index += 1;
    }
  }
  return kept.join(" ");
}

/** Collapse 3+ identical consecutive words: "I I I I" -> "I". Newlines are
 *  kept because a spoken "new line" is a word the cleanup must not swallow. */
export function collapseStutters(text: string): string {
  if (!text) return text;
  return text
    .split(/(\n+)/)
    .map((part) => (/^\n+$/.test(part) ? part : collapseStutterLine(part)))
    .join("");
}

/** Non-filler cleanup that stays on even when filler removal is off. */
export function normalizeTranscriptionOutput(text: string): string {
  return normalizeWhitespace(collapseStutters(text));
}

interface SpokenCommand {
  pattern: RegExp;
  value: string;
  /** Opening marks attach to the word after them, closing marks to the word
   *  before them, so a quoted phrase is `"like this"` rather than `" like this "`. */
  attach?: "before" | "after";
}

/** Longest phrases first so "question mark" never becomes "question" plus a
 *  stray mark. The final entry only fires at the end of a segment: "period" is
 *  also an ordinary English word, while "full stop" is not. */
const SPOKEN_COMMANDS: SpokenCommand[] = [
  { pattern: /\bnew\s+paragraph\b/gi, value: "\n\n" },
  { pattern: /\bnew\s+line\b/gi, value: "\n" },
  { pattern: /\bquestion\s+mark\b/gi, value: "?" },
  { pattern: /\bexclamation\s+(?:mark|point)\b/gi, value: "!" },
  { pattern: /\bfull\s+stop\b/gi, value: "." },
  { pattern: /\bsemicolon\b/gi, value: ";" },
  { pattern: /\bcolon\b/gi, value: ":" },
  { pattern: /\bcomma\b/gi, value: "," },
  { pattern: /\bellipsis\b/gi, value: "..." },
  { pattern: /\bopen\s+paren(?:thesis)?\b/gi, value: "(", attach: "before" },
  { pattern: /\bclose\s+paren(?:thesis)?\b/gi, value: ")", attach: "after" },
  { pattern: /\bopen\s+quote\b/gi, value: '"', attach: "before" },
  { pattern: /\bclose\s+quote\b/gi, value: '"', attach: "after" },
  { pattern: /\bdash\b/gi, value: " - " },
  { pattern: /\s+period\s*$/i, value: "." },
];

const TOKEN_OPEN = "\uE000";
const TOKEN_CLOSE = "\uE001";

/** Turn spoken punctuation words into their marks without touching punctuation
 *  the model already produced. Commands are swapped for private-use tokens
 *  first, so "example.com" keeps its dot and only a spoken "full stop" becomes
 *  sentence punctuation. */
export function applySpokenPunctuation(text: string): string {
  if (!text) return text;
  let result = text;
  SPOKEN_COMMANDS.forEach((command, index) => {
    result = result.replace(command.pattern, `${TOKEN_OPEN}${index}${TOKEN_CLOSE}`);
  });

  result = result.replace(
    new RegExp(`[ \\t]*${TOKEN_OPEN}(\\d+)${TOKEN_CLOSE}[ \\t]*`, "g"),
    (_match, rawIndex: string) => {
      const command = SPOKEN_COMMANDS[Number(rawIndex)];
      const value = command?.value ?? "";
      if (value === "\n" || value === "\n\n") return value;
      if (command?.attach === "before") return ` ${value}`;
      if (command?.attach === "after") return `${value} `;
      if (/^[,.;:!?)\]]/.test(value)) return `${value} `;
      if (/^[([]/.test(value)) return ` ${value}`;
      return ` ${value} `;
    },
  );

  result = normalizeWhitespace(result);
  // A command can start a sentence; the model's next word is lowercase. An
  // opening quote or bracket belongs to the sentence, so look past it.
  return result.replace(
    /(^|[.!?\u2026]\s+|\n+)(["'\u201c\u201d\u2018\u2019([]*)(\p{Ll})/gu,
    (_match, prefix: string, opening: string, letter: string) =>
      `${prefix}${opening}${letter.toUpperCase()}`,
  );
}

export interface DictationTextOptions {
  /** Output language, e.g. "en" or "de-DE". Empty means the model did not say. */
  language?: string;
  removeFillers?: boolean;
  spokenPunctuation?: boolean;
}

/** The full per-segment cleanup pipeline. */
export function cleanDictationText(
  text: string,
  options: DictationTextOptions = {},
): string {
  let result = text;
  if (options.spokenPunctuation) {
    result = applySpokenPunctuation(result);
  }
  if (options.removeFillers) {
    result = removeFillerWords(result, options.language ?? "");
  }
  return normalizeTranscriptionOutput(result);
}

/** Append one transcript to text the user already has, preserving that text's
 *  own whitespace (a draft can end in a newline or a space the user typed).
 *  Leading and trailing newlines are kept: a spoken "new paragraph" is a real
 *  part of the transcript, even when it is the whole segment. */
export function appendTranscriptText(base: string, text: string): string {
  const part = text.trim();
  const leading = /^\n+/.exec(text)?.[0] ?? "";
  const trailing = /\n+$/.exec(text)?.[0] ?? "";
  if (!part) {
    const breaks = leading || trailing;
    if (!breaks || !base) return base;
    return /\n$/.test(base) ? base : `${base}${breaks}`;
  }
  if (!base) return `${part}${trailing}`;
  const breakBefore = leading && !/\n$/.test(base) ? leading : "";
  const before = breakBefore ? base.replace(/[ \t]+$/, "") : base;
  const separator =
    breakBefore || /^[,.;:!?)\]]/.test(part) || /\s$/.test(before) ? "" : " ";
  return `${before}${breakBefore}${separator}${part}${trailing}`;
}

/** Join segment transcripts without putting a space before punctuation that a
 *  segment started with (a spoken "comma" often begins the next segment). */
export function joinTranscriptParts(parts: string[]): string {
  let result = "";
  for (const part of parts) {
    // The speech model punctuates short pauses. A period followed by a
    // lowercase continuation is one of those pauses, not a sentence end.
    if (result.endsWith(".") && /^\s*[a-z]/.test(part)) {
      result = result.slice(0, -1);
    }
    result = appendTranscriptText(result, part);
  }
  return result;
}
