// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


const BACKCHANNEL = new Set([
  "mm", "mmm", "hmm", "hm", "mhm", "mm-hmm", "mmhmm", "uh", "um", "uhm", "umm", "ah", "ahh",
  "oh", "ooh", "huh", "eh", "er", "erm", "hmmm", "uh-huh", "uh-uh",
]);

export function isBackchannel(text: string): boolean {
  const words = text.toLowerCase().replace(/[^\p{L}\-\s]/gu, " ").split(/\s+/).filter(Boolean);
  return words.length > 0 && words.every((w) => BACKCHANNEL.has(w) || /^(m+|h+m+|u+h+|a+h+|o+h+)$/.test(w));
}

// Only ever dropped as the whole transcript.
const STT_NOISE = new Set(["you", "thank you", "thank you very much", "thanks", "thanks for watching", "bye", "the end"]);

export function isSttNoise(text: string): boolean {
  return STT_NOISE.has(text.toLowerCase().replace(/[^\p{L}\s]/gu, " ").trim().replace(/\s+/g, " "));
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

export function isReplyEcho(transcript: string, reply: string): boolean {
  const t = wordsOf(transcript);
  if (t.length === 0 || !reply) return false;
  if (t.length === 1 && ONE_WORD_COMMANDS.has(t[0])) return false;
  // A word or two taken from a closing question ("..., Python or JavaScript?") is the user's answer.
  const asked = reply.trim().endsWith("?") ? (reply.split(/(?<=[.!?\n])\s+/).pop() ?? "") : "";
  if (t.length <= 2 && asked && t.every((w) => wordsOf(asked).includes(w))) return false;
  // Whole words only: "hey" is not inside "they", nor "skip" inside "skipping".
  const r = ` ${wordsOf(reply).join(" ")} `;
  return r.includes(` ${t.join(" ")} `);
}
