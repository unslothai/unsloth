// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Turning assistant text into what the codec actually speaks, and into the chunks
// it speaks one at a time. Dependency-free for the same reason tts-audio-types is:
// these are the only part of the voice path testable without a mic or a GPU, and
// the player hook they used to live in drags in React, auth and asset imports.

const SPEECH_STRIP_RE =
  /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{20E3}\u{FE00}-\u{FE0F}\u{200D}]/gu;

// Normalize text for TTS: some characters derail Orpheus (like the colon it reads
// as a speaker tag) or get voiced literally by any TTS model. Em/en dashes and the
// single-char ellipsis are the main offenders (an em dash breaks the voice), and
// markdown/markup symbols get read out ("asterisk asterisk"). All of these are just
// dropped (replaced with a space, not a comma, so no phantom pauses are inserted);
// smart quotes are normalized. Speech ONLY -- the on-screen chat text keeps everything.
export function stripForSpeech(text: string): string {
  return text
    .replace(SPEECH_STRIP_RE, "")
    .replace(/[^\S\n]*[—–―‒−][^\S\n]*/g, " ") // — – ― ‒ −  -> drop
    .replace(/[^\S\n]*…[^\S\n]*/g, " ") //                 …  -> drop
    .replace(/\.{2,}/g, " ") //                                 ... -> drop
    .replace(/[‐‑]/g, "-") //                          unicode hyphens -> ASCII
    .replace(/[*_`~^|#<>\\{}[\]]/g, " ") //                      markdown / markup -> space
    .replace(/[‘’‚‛]/g, "'") //             smart single quotes
    .replace(/[“”„‟]/g, '"') //             smart double quotes
    // Collapse horizontal whitespace only. Newlines are sentence boundaries to
    // splitIntoSentences, and flattening them here (as this used to) made a
    // multi-paragraph reply one enormous sentence: the loop then waited on a
    // single long synth instead of speaking the first line straight away.
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

// Words whose trailing period ends an abbreviation, not a sentence. Splitting
// there hands the codec a fragment ("Dr.") and starts the next clip mid-thought.
const SPEECH_ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "eg",
  "ie", "approx", "dept", "est", "fig", "no", "vol", "inc", "ltd", "co",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec",
]);

export function splitIntoSentences(text: string): string[] {
  // Horizontal whitespace only: a newline is a boundary the regex below relies on.
  const norm = text
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
  const raw = norm.match(/[^.!?\n]*[.!?]+|\S[^.!?\n]*$|\S[^.!?\n]*(?=\n)/g);
  if (!raw) return [text];
  const out: string[] = [];
  for (const chunk of raw) {
    const part = chunk.trim();
    if (!part) continue;
    const prev = out[out.length - 1];
    if (prev !== undefined) {
      // Quotes and brackets only CLOSE when whitespace or the end follows one.
      // Followed by a letter it opens a quotation, which is a real boundary.
      const closer = /^["')\]]+(\s|$)/.test(part);
      const tail = prev.replace(/["')\]]+$/, "");
      const lastWord = /([A-Za-z]+)\.$/.exec(tail)?.[1];
      const decimal = /\d\.$/.test(tail) && /^\d/.test(part);
      const initial = /(^|\s)[A-Za-z]\.$/.test(tail) && /^[A-Za-z]\./.test(part);
      const abbrev =
        lastWord !== undefined && SPEECH_ABBREVIATIONS.has(lastWord.toLowerCase());
      if (decimal || initial || abbrev || closer) {
        // "3.50" and "D.C." close up; "Dr. Smith" keeps the space it was written with.
        out[out.length - 1] = prev + (decimal || initial || closer ? "" : " ") + part;
        continue;
      }
    }
    out.push(part);
  }
  return out.filter(Boolean);
}
