// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { useAuiState } from "@assistant-ui/react";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { roleplayMetaOf } from "./meta";
import { useCharactersStore } from "./store";
import type { Character, ChatView } from "./types";

/** Per-chat clicked character; holds until the next message. */
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

/** Clicked character, else the last (or next) speaker, else the first character. */
type MessageLike = Parameters<typeof roleplayMetaOf>[0];

function lastSpeakerOf(messages: readonly MessageLike[]): string | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const meta = roleplayMetaOf(messages[i]);
    if (meta?.kind === "reply" || meta?.kind === "turn") return meta.speakerId;
  }
  return null;
}

export function focusedCharacterId(threadId: string, messages: readonly MessageLike[], castIds: string[]): string | null {
  const pin = useStageFocusStore.getState().pins[threadId];
  const pinned = pin && pin.messageCount === messages.length ? pin.characterId : null;
  const last = lastSpeakerOf(messages);
  return [pinned, last, castIds[0]].find((id) => id && castIds.includes(id)) ?? null;
}

export function useStageFocus(threadId: string | null, cast: Character[]): { focus: Character | null; speaking: boolean } {
  const state = useAuiState(
    ({ thread }) => `${thread.isRunning ? 1 : 0}|${thread.messages.length}|${lastSpeakerOf(thread.messages) ?? ""}`,
  );
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

/** Shows a character's picture beside the chat: stage view opens unless a picture view already is. */
export function showOnStage(threadId: string, characterId: string, messageCount: number): void {
  useStageFocusStore.setState((state) => ({
    pins: { ...state.pins, [threadId]: { characterId, messageCount } },
  }));
  const scene = useCharactersStore.getState().scenes[threadId];
  if (chatViewOf(scene?.settings) === "chat") setChatView(threadId, "stage");
}
