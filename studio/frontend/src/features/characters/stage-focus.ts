// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { useAuiState } from "@assistant-ui/react";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { roleplayMetaOf } from "./meta";
import { useCharactersStore } from "./store";
import type { Character, ChatView } from "./types";

/**
 * The character a click put in the picture, per chat. It holds until the chat gets another message,
 * then the picture follows whoever speaks.
 */
interface StagePin {
  characterId: string;
  messageCount: number;
}

export const useStageFocusStore = create<{ pins: Record<string, StagePin> }>(() => ({ pins: {} }));

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => useStageFocusStore.setState({ pins: {} }));
}

export function chatViewOf(settings: { view?: ChatView } | undefined): ChatView {
  return settings?.view ?? "chat";
}

/**
 * Who the picture shows: a character the user clicked, until the next message, otherwise whoever
 * spoke last (or is being asked to), otherwise the first character.
 */
export function useStageFocus(threadId: string | null, cast: Character[]): { focus: Character | null; speaking: boolean } {
  const state = useAuiState(({ thread }) => {
    let lastId: string | null = null;
    for (let i = thread.messages.length - 1; i >= 0; i -= 1) {
      const meta = roleplayMetaOf(thread.messages[i]);
      if (meta?.kind === "reply" || meta?.kind === "turn") {
        lastId = meta.speakerId;
        break;
      }
    }
    return `${thread.isRunning ? 1 : 0}|${thread.messages.length}|${lastId ?? ""}`;
  });
  const pin = useStageFocusStore((s) => (threadId ? s.pins[threadId] : undefined));
  const [running, count, lastId] = state.split("|");
  const pinned = pin && pin.messageCount === Number(count) ? pin.characterId : null;
  const focus =
    cast.find((c) => c.id === pinned) ?? cast.find((c) => c.id === lastId) ?? cast[0] ?? null;
  return { focus, speaking: running === "1" && focus?.id === lastId };
}

/** Switches a chat between the plain chat, stage view and visual novel view; remembered per chat. */
export function setChatView(threadId: string, view: ChatView): void {
  void useCharactersStore
    .getState()
    .patchScene(threadId, { settings: { view } })
    .catch((error) => toast.error(error instanceof Error ? error.message : String(error)));
}

/** Shows a character's picture beside the chat: stage view opens unless a picture view already is. */
export function showOnStage(threadId: string, characterId: string, messageCount: number): void {
  useStageFocusStore.setState((state) => ({
    pins: { ...state.pins, [threadId]: { characterId, messageCount } },
  }));
  const scene = useCharactersStore.getState().scenes[threadId];
  if (chatViewOf(scene?.settings) === "chat") setChatView(threadId, "stage");
}
