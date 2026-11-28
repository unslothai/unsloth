// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type NameSpelling = { source: string; target: string };

const MAX_NAMES_PER_PASSAGE = 12;

// German and Luxembourgish capitalise every noun, so capitals do not mark names there.
const CAPITALISES_NOUNS = new Set(["de", "lb"]);

export function namesAreCapitalised(languageCode: string | null): boolean {
  return languageCode === null || !CAPITALISES_NOUNS.has(languageCode);
}

// English capitalises these, but they are words to translate, not names to keep.
const CALENDAR_WORDS = new Set(
  "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September October November December".split(
    " ",
  ),
);

const WORD = String.raw`\p{Lu}\p{Ll}[\p{Ll}\p{M}]*(?:[-'’]\p{Lu}?\p{Ll}[\p{Ll}\p{M}]*)*`;
const NAME = new RegExp(
  `(?<![\\p{L}\\p{M}\\p{N}])${WORD}(?: ${WORD})*(?![\\p{L}\\p{M}\\p{N}])`,
  "gu",
);
const SENTENCE_START = /(?:^|[.!?…:;]["'”’)\]]*\s+|\n\s*|["“‘([]\s*)$/u;

/** Runs of capitalised words; a lone capitalised word opening a sentence is ordinary text. */
export function findNames(text: string): string[] {
  const names: string[] = [];
  for (const match of text.matchAll(NAME)) {
    const name = match[0];
    if (
      !name.includes(" ") &&
      (CALENDAR_WORDS.has(name) ||
        SENTENCE_START.test(text.slice(0, match.index)))
    )
      continue;
    if (!names.includes(name)) names.push(name);
    if (names.length === MAX_NAMES_PER_PASSAGE) break;
  }
  return names;
}

/** The name as the model wrote it on its own; null when it kept the original or wrote more than a name. */
export function nameSpelling(
  source: string,
  translated: string,
): string | null {
  const target = translated
    .trim()
    .replace(/^["'“”‘’«»「」]+|["'“”‘’«»「」.,;:!?。、，！？।॥۔]+$/gu, "")
    .trim();
  if (!target || target.includes("\n") || target.includes(source)) return null;
  if (target.length > source.length * 4 + 8) return null;
  return target;
}

/** Hy-MT3's model-card Terminology prompt, which a general chat model follows too. */
export function referenceBlock(
  names: readonly NameSpelling[] | undefined,
): string {
  if (!names || names.length === 0) return "";
  const lines = names.map(
    (name) => `${name.source} translates to ${name.target}`,
  );
  return `Reference the following translations:\n${lines.join("\n")}\n\n`;
}

/** For a general chat model, after its instruction: a block ahead of it can read as the text to translate. */
export function nameGuide(names: readonly NameSpelling[] | undefined): string {
  if (!names || names.length === 0) return "";
  const pairs = names.map((name) => `${name.source} as ${name.target}`);
  return ` Write these names exactly like this: ${pairs.join("; ")}.`;
}

const LETTER = /[\p{L}\p{M}\p{N}]/u;
const WORD_TAIL = /[\p{L}\p{M}\p{N}]+$/u;
// Arabic, Hebrew and Amharic write short prepositions and articles onto the next word.
const PROCLITICS = /^(?:[وفبلكا]{1,3}|[בהוכלמש]{1,2}|[በከለየ]|ስለ|እንደ)$/u;

function startsWord(text: string, at: number): boolean {
  const before = WORD_TAIL.exec(text.slice(Math.max(0, at - 4), at))?.[0];
  return !before || (before.length < 4 && PROCLITICS.test(before));
}

/** Adds "(original)" after the first mention of each name; a spelling the model did not use is left alone. */
export function annotateNames(
  text: string,
  names: readonly NameSpelling[],
  annotated: Set<string>,
): string {
  // Longest first, every mention claimed: a surname alone must not match inside its full name.
  const claimed: Array<[number, number]> = [];
  const inserts: Array<[number, string]> = [];
  // Two names spelled alike cannot be told apart in the output, so neither is annotated.
  const shared = (target: string) =>
    names.filter((name) => name.target === target).length > 1;
  const byLength = [...names].sort((a, b) => b.target.length - a.target.length);
  for (const { source, target } of byLength) {
    for (
      let at = text.indexOf(target);
      at >= 0;
      at = text.indexOf(target, at + 1)
    ) {
      const end = at + target.length;
      if (LETTER.test(text[end] ?? "") || !startsWord(text, at)) continue;
      if (claimed.some(([from, to]) => at < to && end > from)) continue;
      claimed.push([at, end]);
      if (annotated.has(source) || shared(target)) continue;
      annotated.add(source);
      if (!text.slice(end).trimStart().startsWith(`(${source})`))
        inserts.push([end, ` (${source})`]);
    }
  }
  let out = text;
  for (const [end, note] of inserts.sort((a, b) => b[0] - a[0]))
    out = out.slice(0, end) + note + out.slice(end);
  return out;
}
