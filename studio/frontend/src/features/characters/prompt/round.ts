// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Turn-taking in a character chat, free of app state so it can be tested on its own. */

import { roleplayMetaOf, roundPlanOf } from "../meta.ts";
import type { Character, ChatScene } from "../types";
import { planReplies } from "./speakers.ts";

export type MessageLike = {
  id: string;
  role: string;
  content: unknown;
  metadata?: unknown;
  status?: { type?: string };
};

/** A reply that was stopped, failed or is waiting on the user, which ends its round. */
export function endedEarly(message: MessageLike): boolean {
  if (message.role !== "assistant") return false;
  const type = message.status?.type;
  if (type === "incomplete" || type === "requires-action") return true;
  const metadata = message.metadata as { custom?: Record<string, unknown> } | undefined;
  return Boolean(metadata?.custom?.incomplete);
}

export function textOf(message: { content: unknown }): string {
  const content = message.content as string | readonly { type: string; text?: string }[];
  if (typeof content === "string") return content;
  return content
    .map((part) => (part.type === "text" ? (part.text ?? "") : ""))
    .filter(Boolean)
    .join("\n\n");
}

export function castOf(scene: ChatScene, characters: Record<string, Character>): Character[] {
  return [...scene.cast]
    .sort((a, b) => a.position - b.position)
    .map((member) => characters[member.characterId])
    .filter((character): character is Character => Boolean(character));
}

/**
 * Who speaks in this run. An explicit request or a turn marker names them; a lone character always
 * answers; otherwise the group plans its replies to the user's message, seeded by that message so a
 * retry keeps the same speaker.
 */
export function resolveSpeaker(input: {
  scene: ChatScene;
  cast: Character[];
  messages: readonly MessageLike[];
  requested?: string | null;
}): { speakerId: string; plan?: string[] } | null {
  const { scene, cast, messages } = input;
  if (cast.length === 0) return null;
  const ids = new Set(cast.map((c) => c.id));
  if (input.requested && ids.has(input.requested)) return { speakerId: input.requested };
  const last = messages[messages.length - 1];
  const lastMeta = roleplayMetaOf(last as never);
  if (lastMeta?.kind === "turn" && ids.has(lastMeta.speakerId)) return { speakerId: lastMeta.speakerId };
  // Only the first member's greeting hangs under the opening; retrying it keeps it theirs.
  if (cast.length === 1 || lastMeta?.kind === "opening") return { speakerId: cast[0].id };
  const muted = new Set(scene.cast.filter((m) => m.muted).map((m) => m.characterId));
  let lastSpeakerId: string | null = null;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const meta = roleplayMetaOf(messages[i] as never);
    if (meta?.kind === "reply") {
      lastSpeakerId = meta.speakerId;
      break;
    }
  }
  const plan = planReplies({
    cast: cast.map((c) => ({ id: c.id, name: c.name, talkativeness: c.talkativeness, muted: muted.has(c.id) })),
    order: scene.settings.replyOrder ?? "natural",
    userText: last ? textOf(last) : "",
    lastSpeakerId,
    allowSelfReplies: scene.settings.allowSelfReplies,
    seed: last?.id ?? "",
  });
  const speakerId = plan[0] ?? cast.find((c) => !muted.has(c.id))?.id ?? cast[0].id;
  return { speakerId, plan: plan.length ? plan : [speakerId] };
}

/** The next group speaker after a finished reply, or null when the round is over. */
export function nextInRound(messages: readonly MessageLike[], scene: ChatScene): string | null {
  if (scene.cast.length < 2) return null;
  let start = messages.length - 1;
  while (start >= 0) {
    const message = messages[start];
    if (message.role === "user" && !roleplayMetaOf(message as never)) break;
    start -= 1;
  }
  if (start < 0) return null;
  const round = messages.slice(start + 1);
  // A stopped round stays over, even after a reply someone asked for by hand.
  if (round.some(endedEarly)) return null;
  const firstReply = round.find((m) => m.role === "assistant");
  const plan = roundPlanOf(firstReply as never);
  if (!plan) return null;
  const spoken = new Set<string>();
  for (const message of round) {
    const meta = roleplayMetaOf(message as never);
    if (meta?.kind === "reply") spoken.add(meta.speakerId);
  }
  const present = new Set(scene.cast.filter((m) => !m.muted).map((m) => m.characterId));
  return plan.find((id) => !spoken.has(id) && present.has(id)) ?? null;
}
