// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReplyOrder } from "../types";

export interface SpeakerCandidate {
  id: string;
  name: string;
  talkativeness: number;
  muted: boolean;
}

/** A small deterministic generator: the same message picks the same speakers on every retry. */
export function seededRandom(seed: string): () => number {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i += 1) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  let state = h >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Titles that open a name but are not what people call the character ("Captain Rhys Calloway" is Rhys).
const TITLES = new Set([
  "captain", "capt", "inspector", "detective", "professor", "prof", "doctor", "dr", "sir", "lady", "lord",
  "dame", "mr", "mrs", "ms", "miss", "mx", "king", "queen", "prince", "princess", "commander", "general",
  "lieutenant", "lt", "sergeant", "sgt", "officer", "agent", "father", "sister", "brother", "master", "madam",
]);

export function isTitle(word: string): boolean {
  return TITLES.has(word.toLowerCase().replace(/\.$/, ""));
}

/** What a character is called in passing: the first word of the name that is not a title. */
export function shortName(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  const word = words.find((w) => !isTitle(w)) ?? words[0] ?? "";
  return word || name.trim();
}

function escape(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Members named in the text, in the order they are first named: full name, short name, surname, or a title only they hold. */
export function mentionedIds(text: string, cast: SpeakerCandidate[]): string[] {
  const words = (member: SpeakerCandidate) => member.name.trim().split(/\s+/).filter(Boolean);
  // A title calls its character only when one character has it ("Captain, you've gone quiet").
  const titleCount = new Map<string, number>();
  for (const member of cast) {
    for (const word of words(member).filter(isTitle)) titleCount.set(word, (titleCount.get(word) ?? 0) + 1);
  }
  const found: { id: string; at: number }[] = [];
  for (const member of cast) {
    const all = words(member);
    const plain = all.filter((word) => !isTitle(word));
    const loose = [member.name.trim(), shortName(member.name), plain.length > 1 ? plain[plain.length - 1] : ""]
      .filter((name, i, list) => name && (i === 0 || name.length >= 3) && list.indexOf(name) === i);
    let at = -1;
    const see = (index: number) => {
      if (index >= 0 && (at === -1 || index < at)) at = index;
    };
    for (const name of loose) {
      see(new RegExp(`(?<![\\p{L}\\p{N}_])${escape(name)}(?![\\p{L}\\p{N}_])`, "iu").exec(text)?.index ?? -1);
    }
    for (const title of all.filter((word) => isTitle(word) && titleCount.get(word) === 1)) {
      // Capitalized and not possessive: "Captain," calls him, "the captain's coat" does not.
      see(new RegExp(`(?<![\\p{L}\\p{N}_])${escape(title)}(?![\\p{L}\\p{N}_'\u2019])`, "u").exec(text)?.index ?? -1);
    }
    if (at >= 0) found.push({ id: member.id, at });
  }
  return found.sort((a, b) => a.at - b.at).map((entry) => entry.id);
}

/**
 * Who answers a user message, in order.
 * natural: members named in the message first, then each other member by a talkativeness roll, at least one.
 * list: every unmuted member in cast order.
 * manual ("one at a time"): exactly one, the first named member or else the member after the last speaker.
 */
export function planReplies(input: {
  cast: SpeakerCandidate[];
  order: ReplyOrder;
  userText: string;
  lastSpeakerId: string | null;
  allowSelfReplies?: boolean;
  seed: string;
}): string[] {
  const active = input.cast.filter((member) => !member.muted);
  if (active.length === 0) return [];
  if (active.length === 1) return [active[0].id];
  const random = seededRandom(input.seed);
  const mentioned = mentionedIds(input.userText, active);
  if (input.order === "list") return active.map((member) => member.id);
  if (input.order === "manual") {
    if (mentioned.length) return [mentioned[0]];
    const last = active.findIndex((member) => member.id === input.lastSpeakerId);
    return [active[(last + 1) % active.length].id];
  }
  // Named members answer, else one by talkativeness; at most one extra voice chimes in.
  const canFollow = (member: SpeakerCandidate) => input.allowSelfReplies || member.id !== input.lastSpeakerId;
  const pickWeighted = (pool: SpeakerCandidate[]): SpeakerCandidate | null => {
    if (pool.length === 0) return null;
    const weights = pool.map((member) => Math.max(0.05, Math.min(1, member.talkativeness)));
    let roll = random() * weights.reduce((sum, weight) => sum + weight, 0);
    for (let i = 0; i < pool.length; i += 1) {
      roll -= weights[i];
      if (roll < 0) return pool[i];
    }
    return pool[pool.length - 1];
  };
  const queue = [...mentioned];
  if (queue.length === 0) {
    const lead = pickWeighted(active.filter(canFollow)) ?? pickWeighted(active);
    if (lead) queue.push(lead.id);
  }
  // A named answer is rarely interrupted (a quarter of the usual chance); an open line invites more.
  const joinChance = mentioned.length ? 0.25 : 0.5;
  const rest = active.filter((member) => !queue.includes(member.id));
  const extra = pickWeighted(rest);
  if (extra && queue.length < 2 && random() < Math.min(1, extra.talkativeness) * joinChance) queue.push(extra.id);
  return queue;
}
