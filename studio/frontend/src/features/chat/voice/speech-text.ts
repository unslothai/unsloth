// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sentence splitting for read aloud; mirrors voice mode's splitter (#14084).

// A trailing period on these ends an abbreviation, not a sentence ("Dr.").
const SPEECH_ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "eg",
  "ie", "approx", "dept", "est", "fig", "no", "vol", "inc", "ltd", "co",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec",
]);

const SENTENCE_FINAL_ABBREVIATIONS = new Set(["etc", "inc", "ltd", "co", "jr", "sr"]);

export function splitIntoSentences(text: string): string[] {
  // Horizontal whitespace only: a newline is a boundary the regex below relies on.
  const norm = text
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
  const raw = norm.match(/[^.!?\n]*[.!?]+|\S[^.!?\n]*$|\S[^.!?\n]*(?=\n)/g);
  if (!raw) return [text];
  const out: string[] = [];
  let at = 0;
  for (const chunk of raw) {
    const start = norm.indexOf(chunk, at);
    at = start + chunk.length;
    let part = chunk.trim();
    // Whether a space or newline came before it: "2.5" has none, "in 2020. 30 people" has one.
    const spaced = /^\s/.test(chunk) || (start > 0 && /\s/.test(norm[start - 1] ?? ""));
    if (!part) continue;
    const prev = out[out.length - 1];
    if (prev !== undefined) {
      // A quote or bracket followed by whitespace closes the previous sentence.
      const closer = /^(["')\]]+)(?:\s|$)/.exec(part);
      if (closer) {
        out[out.length - 1] = prev + closer[1];
        part = part.slice(closer[1].length).trim();
        if (!part) continue;
        out.push(part);
        continue;
      }
      const tail = prev.replace(/["')\]]+$/, "");
      const lastWord = /([A-Za-z]+)\.$/.exec(tail)?.[1];
      const decimal = /\d\.$/.test(tail) && !spaced && /^\d/.test(part);
      // "J. Smith", "John F. Kennedy" are names; "pick option A. Then" is a boundary.
      const nameInitial =
        /(?:^|(?:^|\s)(?:[A-Z][a-z]+|[A-Z]\.)\s)[A-Z]\.$/.test(tail) && /^[A-Z][a-z]/.test(part);
      const initial = /(^|\s)[A-Za-z]\.$/.test(tail) && /^[A-Za-z]\./.test(part);
      // "llama.cpp", "config.json": a dot with a letter right after it is part of a name.
      const dotted = /[A-Za-z\d]\.$/.test(tail) && !spaced && /^[A-Za-z]/.test(part);
      const word = lastWord?.toLowerCase();
      // "etc. Then I left": these also end sentences, so only a lowercase word or digit continues one.
      const abbrev =
        word !== undefined &&
        SPEECH_ABBREVIATIONS.has(word) &&
        (!SENTENCE_FINAL_ABBREVIATIONS.has(word) || /^[a-z\d]/.test(part));
      // "The U.S. is large", "e.g. the docs": a lowercase word after an initialism continues it.
      const initialism = /(?:^|\s)(?:[A-Za-z]\.){2,}$/.test(tail) && /^[a-z]/.test(part);
      // ", the docs" after "e.g." can't start a sentence.
      const continues = /^[,;:]/.test(part);
      if (decimal || initial || dotted || abbrev || initialism || continues || nameInitial) {
        // "3.50" and "D.C." close up; "Dr. Smith" keeps the space it was written with.
        out[out.length - 1] = prev + (decimal || (initial && !spaced) || dotted || continues ? "" : " ") + part;
        continue;
      }
    }
    out.push(part);
  }
  return out.filter(Boolean);
}
