// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatScene } from "./types";

export interface ChatGateState {
  chatIndex: Record<string, string[]>;
  indexLoaded: boolean;
  scenes: Record<string, ChatScene | null>;
}

/**
 * Selector: whether a thread may be a character chat, so ordinary chats skip the scene fetch.
 * Unknown (true) until the chat index has loaded, unless the thread is already known either way.
 */
export function isCharacterChat(threadId: string | null, messagesHaveRoleplayMeta: boolean) {
  return (state: ChatGateState): boolean => {
    if (!threadId) return false;
    if (messagesHaveRoleplayMeta || threadId in state.chatIndex) return true;
    if (threadId in state.scenes) return state.scenes[threadId] !== null;
    return !state.indexLoaded;
  };
}
