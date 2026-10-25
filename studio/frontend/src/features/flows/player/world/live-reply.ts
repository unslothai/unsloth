// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** An NPC line still being written, shown as it streams in. */
export interface LiveReply {
  npcId: string;
  generation: number;
  text: string;
}

/** A piece whose `generation` is behind `current` predates the last save load and is dropped. */
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
