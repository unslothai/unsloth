// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice-typing moderation. This blocklist is deliberately scoped to strong
// profanity and slurs: a dictated sentence must not be inserted, saved, copied
// or kept in history when it contains abusive language. Composer dictation is
// intentionally not routed through this check.

const ABUSIVE_WORDS: ReadonlySet<string> = new Set([
  "asshole",
  "arsehole",
  "bastard",
  "bitch",
  "bitches",
  "bollocks",
  "bullshit",
  "cock",
  "cunt",
  "dick",
  "dickhead",
  "douchebag",
  "fag",
  "faggot",
  "fuck",
  "fucked",
  "fucker",
  "fucking",
  "motherfucker",
  "nigga",
  "nigger",
  "prick",
  "pussy",
  "retard",
  "retarded",
  "shit",
  "shitty",
  "slut",
  "twat",
  "wanker",
  "whore",
  "chink",
  "coon",
  "dyke",
  "kike",
  "paki",
  "raghead",
  "spic",
  "towelhead",
  "tranny",
  "wetback",
]);

const ABUSIVE_PHRASES: readonly string[] = [
  "piece of shit",
  "son of a bitch",
];

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replaceAll("0", "o")
    .replaceAll("1", "i")
    .replaceAll("3", "e")
    .replaceAll("4", "a")
    .replaceAll("5", "s")
    .replaceAll("7", "t")
    .replaceAll("@", "a")
    .replaceAll("$", "s")
    .replace(/[^\p{L}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function collapseRepeats(word: string): string {
  return word.replace(/(.)\1+/gu, "$1");
}

/** True when the transcript contains a strong abusive word or slur. */
export function containsAbusiveLanguage(text: string): boolean {
  if (!text.trim()) return false;
  const normalized = normalize(text);
  for (const phrase of ABUSIVE_PHRASES) {
    if (normalized.includes(phrase)) return true;
  }
  for (const word of normalized.split(" ")) {
    if (ABUSIVE_WORDS.has(word) || ABUSIVE_WORDS.has(collapseRepeats(word))) {
      return true;
    }
  }
  return false;
}
