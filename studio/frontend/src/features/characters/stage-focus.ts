// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { useAuiState } from "@assistant-ui/react";
import { create } from "zustand";
import { chatViewOf, setChatView } from "./chat-view";
import { lastSpeakerId, type MessageLike } from "./meta";
import { useCharactersStore } from "./store";
import type { Character } from "./types";

/** Per-chat clicked character; holds until the next message. */
interface StagePin {
  characterId: string;
  messageCount: number;
}

export const useStageFocusStore = create<{ pins: Record<string, StagePin> }>(() => ({ pins: {} }));

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => useStageFocusStore.setState({ pins: {} }));
}

/** Clicked character (until the next message), else the last speaker, else the first. */
function resolveFocus(
  pin: StagePin | undefined,
  messageCount: number,
  lastId: string | null,
  castIds: readonly string[],
): string | null {
  const pinned = pin && pin.messageCount === messageCount ? pin.characterId : null;
  return [pinned, lastId, castIds[0]].find((id) => id && castIds.includes(id)) ?? null;
}

function focusedCharacterId(threadId: string, messages: readonly MessageLike[], castIds: string[]): string | null {
  const pin = useStageFocusStore.getState().pins[threadId];
  return resolveFocus(pin, messages.length, lastSpeakerId(messages, true), castIds);
}

export function useStageFocus(threadId: string | null, cast: Character[]): { focus: Character | null; speaking: boolean } {
  const running = useAuiState(({ thread }) => thread.isRunning);
  const count = useAuiState(({ thread }) => thread.messages.length);
  const lastId = useAuiState(({ thread }) => lastSpeakerId(thread.messages, true));
  const pin = useStageFocusStore((s) => (threadId ? s.pins[threadId] : undefined));
  const focusId = resolveFocus(pin, count, lastId, cast.map((c) => c.id));
  const focus = cast.find((c) => c.id === focusId) ?? null;
  return { focus, speaking: running && focus?.id === lastId };
}

/** Portrait click: show the character, or close stage view if it already shows them. */
export function toggleOnStage(threadId: string, characterId: string, messages: readonly MessageLike[], castIds: string[]): void {
  const scene = useCharactersStore.getState().scenes[threadId];
  if (chatViewOf(scene?.settings) === "stage" && focusedCharacterId(threadId, messages, castIds) === characterId) {
    useStageFocusStore.setState((state) => {
      const { [threadId]: _closed, ...pins } = state.pins;
      return { pins };
    });
    setChatView(threadId, "chat");
    return;
  }
  showOnStage(threadId, characterId, messages.length);
}

/** Shows a character's avatar beside the chat: stage view opens unless an avatar view already is. */
export function showOnStage(threadId: string, characterId: string, messageCount: number): void {
  useStageFocusStore.setState((state) => ({
    pins: { ...state.pins, [threadId]: { characterId, messageCount } },
  }));
  const scene = useCharactersStore.getState().scenes[threadId];
  if (chatViewOf(scene?.settings) === "chat") setChatView(threadId, "stage");
}
