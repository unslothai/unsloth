// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { usePinnedChatsStore } from "../stores/pinned-chats-store";
import { chatHistoryClearBoundary } from "./chat-history-clear-boundary";
import { clearStoredChats, countStoredChats } from "./chat-history-storage";
import { requestPromptQueueStop } from "./prompt-queue-boundary";
import { stopChatThread } from "./stop-chat-thread";

export const countAllChats = countStoredChats;

export async function clearAllChats(options: { deleteFiles?: boolean } = {}) {
  // Advance before any await so a late initializer cannot recreate a chat after the clear.
  chatHistoryClearBoundary.advance();
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
  // Snapshot first: a chat another client creates and pins mid-clear survives it.
  const pinnedBefore = usePinnedChatsStore.getState().pinnedIds;
  const result = await clearStoredChats(options);
  if (result.backend === "cleared") {
    // Unpin what was cleared, so a never-imported comparison's queued pin cannot block later saves.
    const failed = new Set(result.failedThreadIds);
    const removed = pinnedBefore.filter((id) => !failed.has(id));
    if (removed.length > 0) usePinnedChatsStore.getState().setPinned(removed, false);
  }
  return result;
}
