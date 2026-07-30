// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { clearStoredChats, countStoredChats } from "./chat-history-storage";
import { requestPromptQueueStop } from "./prompt-queue-boundary";
import { stopChatThread } from "./stop-chat-thread";

export const countAllChats = countStoredChats;

export async function clearAllChats() {
  const { runningByThreadId, cancelByThreadId, serverCancelByThreadId } =
    useChatRuntimeStore.getState();
  const activeThreadIds = new Set([
    ...Object.keys(runningByThreadId),
    ...Object.keys(cancelByThreadId),
    ...Object.keys(serverCancelByThreadId),
  ]);
  for (const threadId of activeThreadIds) {
    stopChatThread(threadId);
  }
  requestPromptQueueStop();
  return await clearStoredChats();
}
