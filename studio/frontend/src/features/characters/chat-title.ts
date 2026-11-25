// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- The chat barrel loads the chat page, which imports this feature.
import { getStoredChatThread, updateStoredChatThread } from "@/features/chat/utils/chat-history-storage";
import type { Character, ChatScene } from "./types";

/** The title a character chat starts with; the same as chat_title in backend/core/characters/chats.py. */
export function castTitle(names: readonly string[]): string {
  if (names.length === 0) return "Character chat";
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")} & ${names[names.length - 1]}`;
}

/** The cast's names in order, or null when one is not loaded (its title cannot be told apart then). */
export function castNames(scene: ChatScene | null | undefined, characters: Record<string, Character>): string[] | null {
  if (!scene) return null;
  const names = [...scene.cast].sort((a, b) => a.position - b.position).map((m) => characters[m.characterId]?.name);
  return names.every((name): name is string => typeof name === "string") ? names : null;
}

/** After the cast changes, a title still naming the old cast names the new one; a title the user chose stays. */
export async function retitleForCast(threadId: string, before: string[] | null, after: string[] | null): Promise<void> {
  if (!before || !after) return;
  const previous = castTitle(before);
  const next = castTitle(after);
  if (previous === next) return;
  const thread = await getStoredChatThread(threadId);
  if (thread?.title !== previous) return;
  await updateStoredChatThread(threadId, { title: next });
}
