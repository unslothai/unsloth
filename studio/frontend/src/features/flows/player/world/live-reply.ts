// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** An NPC line still being written, shown as it streams in. */
export interface LiveReply {
  npcId: string;
  generation: number;
  text: string;
}

/**
 * Add a streamed piece of an NPC's reply. A piece from a request made before the last save load
 * (`generation` behind `current`) belongs to the old game and is dropped.
 */
export function appendReplyDelta(
  live: LiveReply | null,
  npcId: string,
  generation: number,
  current: number,
  delta: string,
): LiveReply | null {
  if (generation !== current || !delta) return live;
  if (!live || live.npcId !== npcId || live.generation !== generation) {
    return { npcId, generation, text: delta };
  }
  return { ...live, text: live.text + delta };
}

/** The reply failed because no chat model can answer here (none loaded, not downloaded, or none picked). */
export function isNoChatModelError(message: string): boolean {
  return /no (chat )?model is loaded|could not load|not downloaded|needs? a chat model|no model (is )?(picked|selected)/i.test(
    message,
  );
}

const OFFLINE_LINES = [
  "Mm. Not much to say right now.",
  "Ask me again later, traveller.",
  "Lovely day for a walk, isn't it?",
];

/**
 * What a villager says when no chat model can answer (a baked world played offline): their greeting first,
 * then a few stock lines, so the conversation still reads as theirs instead of an error.
 */
export function offlineLine(greeting: string | undefined, npcTurnsSoFar: number): string {
  if (npcTurnsSoFar === 0 && greeting?.trim()) return greeting.trim();
  return OFFLINE_LINES[npcTurnsSoFar % OFFLINE_LINES.length];
}

export const OFFLINE_NOTE =
  "Talking needs a chat model: load one in Chat, or pick one on the NPC Reply block.";
