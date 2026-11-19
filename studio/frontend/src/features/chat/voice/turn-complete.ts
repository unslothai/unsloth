// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


// Only when the turn judge cannot answer (none configured, loading, timed out): a fragment ending on
// a dangling word or comma waits for the rest instead of starting a reply.
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

export function looksUnfinished(text: string): boolean {
  const t = text.trim();
  if (!t) return false;
  if (/[,;:\-–—]$|\.\.\.$|…$/.test(t)) return true;
  if (/[.!?]["')\]]?$/.test(t)) return false;
  const words = wordsOf(t);
  const last = words[words.length - 1] ?? "";
  if (DANGLING.has(last)) return true;
  const tail = words.slice(-2).join(" ");
  return tail === "you know" || tail === "i mean" || tail === "kind of" || tail === "sort of";
}

const BACKCHANNEL = new Set([
  "mm", "mmm", "hmm", "hm", "mhm", "mm-hmm", "mmhmm", "uh", "um", "uhm", "umm", "ah", "ahh",
  "oh", "ooh", "huh", "eh", "er", "erm", "hmmm", "uh-huh", "uh-uh",
]);

export function isBackchannel(text: string): boolean {
  const words = text.toLowerCase().replace(/[^\p{L}\-\s]/gu, " ").split(/\s+/).filter(Boolean);
  return words.length > 0 && words.every((w) => BACKCHANNEL.has(w) || /^(m+|h+m+|u+h+|a+h+|o+h+)$/.test(w));
}

const plainSpeech = (t: string) => t.toLowerCase().replace(/[^\p{L}\s]/gu, " ").trim().replace(/\s+/g, " ");

const ACKNOWLEDGEMENTS = new Set([
  "okay", "ok", "right", "sure", "alright", "all right", "i see", "got it", "wow", "really", "cool",
  "nice", "true", "interesting", "makes sense", "go on",
]);
const ANSWER_WORDS = new Set(["yes", "yeah", "yep"]);

/** Whole transcript only: "okay, stop" is a request. */
export function isAcknowledgement(transcript: string, reply: string): boolean {
  const said = plainSpeech(transcript);
  if (ACKNOWLEDGEMENTS.has(said)) return true;
  return ANSWER_WORDS.has(said) && !reply.trim().endsWith("?");
}

// Only ever dropped as the whole transcript.
const STT_NOISE = new Set(["you", "thank you", "thank you very much", "thanks", "thanks for watching", "bye", "the end"]);

export function isSttNoise(text: string): boolean {
  return STT_NOISE.has(plainSpeech(text));
}

export function joinTurn(held: string, next: string): string {
  const a = held.trim();
  const b = next.trim();
  if (!a) return b;
  if (!b) return a;
  const lowered = /^[A-Z][a-z]/.test(b) && !/^I\b/.test(b) ? b[0].toLowerCase() + b.slice(1) : b;
  return `${a.replace(/[,;:\-–—]$|\.\.\.$|…$/, "")} ${lowered}`;
}

const wordsOf = (t: string) =>
  t.toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, " ").split(/\s+/).filter(Boolean);

// A lone command word is the user's even when the reply says it; any other single reply word is echo leak.
const ONE_WORD_COMMANDS = new Set([
  "stop", "wait", "pause", "cancel", "hold", "enough", "quiet", "no", "nope", "yes", "yeah", "yep",
]);

/** A word or two taken from the reply's closing question ("..., Python or JavaScript?"): the user's answer. */
export function answersClosingQuestion(transcript: string, reply: string): boolean {
  const t = wordsOf(transcript);
  const asked = reply.trim().endsWith("?") ? (reply.split(/(?<=[.!?\n])\s+/).pop() ?? "") : "";
  return t.length > 0 && t.length <= 2 && Boolean(asked) && t.every((w) => wordsOf(asked).includes(w));
}

export function isReplyEcho(transcript: string, reply: string): boolean {
  const t = wordsOf(transcript);
  if (t.length === 0 || !reply) return false;
  if (t.length === 1 && ONE_WORD_COMMANDS.has(t[0])) return false;
  if (answersClosingQuestion(transcript, reply)) return false;
  // Whole words only: "hey" is not inside "they", nor "skip" inside "skipping".
  const r = ` ${wordsOf(reply).join(" ")} `;
  return r.includes(` ${t.join(" ")} `);
}
