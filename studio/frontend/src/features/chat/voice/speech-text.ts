// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A trailing period on these ends an abbreviation, not a sentence ("Dr.").
const SPEECH_ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "eg",
  "ie", "approx", "dept", "est", "fig", "no", "vol", "inc", "ltd", "co",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec",
]);

const SENTENCE_FINAL_ABBREVIATIONS = new Set(["etc", "inc", "ltd", "co", "jr", "sr"]);

const speakable = (s: string) => /[\p{L}\p{N}]/u.test(s);

export function splitIntoSentences(text: string): string[] {
  const norm = text
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
  // A sentence ends at .!? (plus closing quotes) only before whitespace or the end, so "3.5",
  // ".env", "./train.py" and "D.C." stay whole; a line without one ends at the newline.
  const raw = norm.match(/\S[^\n]*?(?:[.!?]+["'”’»)\]]*(?=\s|$)|(?=\n|$))/g);
  if (!raw) return [text];
  const out: string[] = [];
  let lead = "";
  for (let part of raw) {
    if (lead) {
      part = `${lead} ${part}`;
      lead = "";
    }
    const prev = out[out.length - 1];
    // Punctuation alone ("...", "--") is no request of its own.
    if (!speakable(part)) {
      if (prev === undefined) lead = part;
      else out[out.length - 1] = `${prev} ${part}`;
      continue;
    }
    if (prev !== undefined) {
      const tail = prev.replace(/["'”’»)\]]+$/, "");
      const word = /([A-Za-z]+)\.$/.exec(tail)?.[1]?.toLowerCase();
      // "J. Smith", "John F. Kennedy" are names; "pick option A. Then" is a boundary.
      const nameInitial =
        /(?:^|(?:^|\s)(?:[A-Z][a-z]+|[A-Z]\.)\s)[A-Z]\.$/.test(tail) && /^[A-Z][a-z]/.test(part);
      const initial = /(^|\s)[A-Za-z]\.$/.test(tail) && /^[A-Za-z]\./.test(part);
      // "etc. Then I left": these also end sentences, so only a lowercase word or digit continues one.
      const abbrev =
        word !== undefined &&
        SPEECH_ABBREVIATIONS.has(word) &&
        (!SENTENCE_FINAL_ABBREVIATIONS.has(word) || /^[a-z\d]/.test(part));
      // "The U.S. is large", "e.g. the docs": a lowercase word after an initialism continues it.
      const initialism = /(?:^|\s)(?:[A-Za-z]\.){2,}$/.test(tail) && /^[a-z]/.test(part);
      if (nameInitial || initial || abbrev || initialism) {
        out[out.length - 1] = `${prev} ${part}`;
        continue;
      }
    }
    out.push(part);
  }
  if (lead) out.push(lead);
  return out;
}
