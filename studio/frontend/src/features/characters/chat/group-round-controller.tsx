// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui, useAuiState } from "@assistant-ui/react";
import { useEffect, useRef } from "react";
import { nextInRound } from "../prompt/round";
import { useCharactersStore } from "../store";
import { registerAsk, useAskToReply } from "./ask-bridge";
import { useCharacterChat } from "./character-chat-context";

// Chats whose next group speaker is scheduled; a queued prompt waits for that speaker to start.
const advancing = new Set<string>();

/** Whether a group round in this chat is about to continue on screen (see GroupRoundController). */
export function groupRoundPending(threadId: string | null | undefined): boolean {
  return Boolean(threadId && advancing.has(threadId));
}

/** Continues a group round after a reply finishes live; a chat switch, stop or failure ends it. */
export function GroupRoundController() {
  const aui = useAui();
  const threadId = useCharacterChat()?.threadId ?? null;
  // The runtime's own thread: it follows the route a render later, and its isRunning belongs to it.
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const running = useAuiState(({ thread }) => thread.isRunning);
  const ask = useAskToReply();
  // Keyed by thread: a switch is not a finish, and a reply already streaming when the chat opens is seen finishing.
  const seen = useRef<{ threadId: string | null; running: boolean }>({ threadId: null, running: false });

  useEffect(() => {
    if (mainThreadId !== threadId) {
      seen.current = { threadId: null, running: false };
      return;
    }
    const finished = seen.current.threadId === threadId && seen.current.running && !running;
    seen.current = { threadId, running };
    if (!finished || !threadId) return;
    const scene = useCharactersStore.getState().scenes[threadId];
    if (!scene || scene.cast.length < 2) return;
    const messages = aui.thread().getState().messages;
    if (messages[messages.length - 1]?.role !== "assistant") return;
    const next = nextInRound(messages, scene);
    if (!next) return;
    // Let the finished reply settle into history before the next run reads it.
    const name = useCharactersStore.getState().characters[next]?.name ?? "";
    advancing.add(threadId);
    const timer = window.setTimeout(() => {
      advancing.delete(threadId);
      if (aui.threads().getState().mainThreadId === threadId && !aui.thread().getState().isRunning) ask(next, name);
    }, 350);
    // Leaving the chat ends the round here, so it must not hold the queue either.
    return () => {
      window.clearTimeout(timer);
      advancing.delete(threadId);
    };
  }, [running, threadId, mainThreadId, aui, ask]);

  useEffect(
    () => (threadId ? registerAsk(threadId, ask, () => aui.thread().getState().messages) : undefined),
    [threadId, ask, aui],
  );

  return null;
}
