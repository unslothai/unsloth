// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether a transcript reads as a finished turn. The energy VAD cannot tell a
// pause to think from the end of a sentence, so the text decides whether the
// loop sends now or holds for the rest. Free of app imports for the node tests.

// Words a sentence almost never ends on: fillers, connectors, articles, the
// prepositions that take an object. "Could you tell me just like" ends on one.
const DANGLING = new Set([
  "um", "uh", "uhm", "erm", "hmm", "like", "just", "so", "and", "but", "or", "nor",
  "because", "cause", "cuz", "then", "than", "that", "which", "who", "whom", "whose",
  "if", "when", "while", "where", "whether", "the", "a", "an", "to", "of", "with",
  "about", "for", "from", "into", "onto", "at", "by", "in", "on", "as", "per", "versus",
  "vs", "my", "your", "his", "her", "its", "our", "their", "this", "these", "those",
  "some", "any", "every", "each", "very", "really", "basically", "actually", "kinda",
  "sorta", "mean", "know", "gonna", "wanna", "is", "are", "was", "were", "be", "been",
  "being", "do", "does", "did", "have", "has", "had", "can", "could", "would", "should",
  "will", "shall", "may", "might", "must", "not", "also", "maybe", "especially",
]);

/** True when the turn looks cut off mid-thought and the loop should keep listening. */
export function looksUnfinished(text: string): boolean {
  const t = text.trim();
  if (!t) return false;
  // A trailing comma, dash or ellipsis is the transcriber saying the same thing.
  if (/[,;:\-–—]$|\.\.\.$|…$/.test(t)) return true;
  // Terminal punctuation is a finished sentence, whatever the last word.
  if (/[.!?]["')\]]?$/.test(t)) return false;
  const words = t.toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, " ").split(/\s+/).filter(Boolean);
  const last = words[words.length - 1] ?? "";
  if (DANGLING.has(last)) return true;
  // "you know" / "I mean" as a tail.
  const tail = words.slice(-2).join(" ");
  return tail === "you know" || tail === "i mean" || tail === "kind of" || tail === "sort of";
}

// Breath, a hum or a throat-clear that cleared the speech gate; Parakeet writes
// them as words. Not a turn, not a fragment: dropped, the mic keeps listening.
const BACKCHANNEL = new Set([
  "mm", "mmm", "hmm", "hm", "mhm", "mm-hmm", "mmhmm", "uh", "um", "uhm", "umm", "ah", "ahh",
  "oh", "ooh", "huh", "eh", "er", "erm", "hmmm", "uh-huh", "uh-uh",
]);

/** True when the transcript is only non-lexical sounds. "Yeah" and "no" are words and stay. */
export function isBackchannel(text: string): boolean {
  const words = text.toLowerCase().replace(/[^\p{L}\-\s]/gu, " ").split(/\s+/).filter(Boolean);
  return words.length > 0 && words.every((w) => BACKCHANNEL.has(w) || /^(m+|h+m+|u+h+|a+h+|o+h+)$/.test(w));
}

/** Join a held fragment and the speech that followed it into one turn. */
export function joinTurn(held: string, next: string): string {
  const a = held.trim();
  const b = next.trim();
  if (!a) return b;
  if (!b) return a;
  // STT capitalises each clip; the continuation is mid-sentence.
  const lowered = /^[A-Z][a-z]/.test(b) && !/^I\b/.test(b) ? b[0].toLowerCase() + b.slice(1) : b;
  return `${a.replace(/[,;:\-–—]$|\.\.\.$|…$/, "")} ${lowered}`;
}
