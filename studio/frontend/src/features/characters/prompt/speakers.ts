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

function escape(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Members named in the text, in the order they are first named. Whole name or first word (3+ letters). */
export function mentionedIds(text: string, cast: SpeakerCandidate[]): string[] {
  const found: { id: string; at: number }[] = [];
  for (const member of cast) {
    const names = [member.name.trim()];
    const first = member.name.trim().split(/\s+/)[0];
    if (first && first.length >= 3 && first !== names[0]) names.push(first);
    let at = -1;
    for (const name of names) {
      if (!name) continue;
      const match = new RegExp(`(?<![\\p{L}\\p{N}_])${escape(name)}(?![\\p{L}\\p{N}_])`, "iu").exec(text);
      if (match && (at === -1 || match.index < at)) at = match.index;
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
  const queue = [...mentioned];
  const others = active.filter(
    (member) =>
      !queue.includes(member.id) &&
      (input.allowSelfReplies || member.id !== input.lastSpeakerId || mentioned.length === 0),
  );
  // Shuffle so equal talkativeness does not always favor cast order.
  for (let i = others.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [others[i], others[j]] = [others[j], others[i]];
  }
  for (const member of others) {
    if (random() < Math.min(1, Math.max(0, member.talkativeness))) queue.push(member.id);
  }
  if (queue.length === 0) {
    const pool = active.filter((member) => member.id !== input.lastSpeakerId);
    const from = pool.length ? pool : active;
    queue.push(from[Math.floor(random() * from.length)].id);
  }
  // A reply right after itself only when allowed or explicitly named.
  if (!input.allowSelfReplies && queue.length > 1 && queue[0] === input.lastSpeakerId && !mentioned.includes(queue[0])) {
    queue.push(queue.shift()!);
  }
  return queue;
}
