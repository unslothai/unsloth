// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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
  const words = t.toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, " ").split(/\s+/).filter(Boolean);
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

export function joinTurn(held: string, next: string): string {
  const a = held.trim();
  const b = next.trim();
  if (!a) return b;
  if (!b) return a;
  const lowered = /^[A-Z][a-z]/.test(b) && !/^I\b/.test(b) ? b[0].toLowerCase() + b.slice(1) : b;
  return `${a.replace(/[,;:\-–—]$|\.\.\.$|…$/, "")} ${lowered}`;
}
