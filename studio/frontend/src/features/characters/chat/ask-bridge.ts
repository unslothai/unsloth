// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui, type CreateAppendMessage, type ThreadMessage } from "@assistant-ui/react";
import { useCallback } from "react";
import { create } from "zustand";
import { CHARACTER_SPEAKER_RUN_KEY } from "../chat-runtime";

type Ask = (characterId: string, name: string) => void;
type Messages = () => readonly ThreadMessage[];

interface Bridge {
  threadId: string | null;
  ask: Ask | null;
  messages: Messages | null;
}

/** Lets UI outside the chat runtime (the side panel) ask a character to reply or read the branch on screen. */
export const useAskBridge = create<Bridge>(() => ({ threadId: null, ask: null, messages: null }));

export function registerAsk(threadId: string, ask: Ask, messages: Messages): () => void {
  useAskBridge.setState({ threadId, ask, messages });
  return () => {
    if (useAskBridge.getState().ask === ask) useAskBridge.setState({ threadId: null, ask: null, messages: null });
  };
}

/** The bridge's ask for this chat, or null while another chat (or none) is on screen. */
export function useBridgedAsk(threadId: string | null | undefined): Ask | null {
  return useAskBridge((s) => (threadId && s.threadId === threadId ? s.ask : null));
}

/** Appends a hidden turn marker that asks one character to speak, then runs it. */
export function useAskToReply(): Ask {
  const aui = useAui();
  return useCallback(
    (characterId: string, name: string) => {
      const thread = aui.thread();
      if (thread.getState().isRunning) return;
      const message: CreateAppendMessage = {
        role: "user",
        // Persisted and exported as is; the prompt builder replaces it with its own instruction.
        content: [{ type: "text", text: `[${name}'s turn]` }],
        metadata: { custom: { roleplay: { kind: "turn", speakerId: characterId } } },
        runConfig: { custom: { [CHARACTER_SPEAKER_RUN_KEY]: characterId } },
        createdAt: new Date(),
      };
      void thread.append(message);
    },
    [aui],
  );
}
