// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBookmarkedTurnsStore } from "../stores/bookmarked-turns-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { usePinnedChatsStore } from "../stores/pinned-chats-store";
import { chatHistoryClearBoundary } from "./chat-history-clear-boundary";
import {
  clearStoredChats,
  countStoredChats,
  listStoredChatThreads,
} from "./chat-history-storage";
import { clearComposerDraft } from "./composer-draft";
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
  // Only finds Compare pairs; an unreachable backend must not stop the clear.
  const threadsBefore = await listStoredChatThreads({
    includeArchived: true,
  }).catch(() => []);
  const result = await clearStoredChats(options);
  const deleted = new Set(result.deletedThreadIds);
  for (const id of deleted) clearComposerDraft(id);
  // A partially deleted comparison still owns its draft.
  const pairs = new Map<string, string[]>();
  for (const thread of threadsBefore) {
    if (!thread.pairId) continue;
    const ids = pairs.get(thread.pairId) ?? [];
    ids.push(thread.id);
    pairs.set(thread.pairId, ids);
  }
  for (const [pairId, ids] of pairs) {
    if (ids.every((id) => deleted.has(id))) clearComposerDraft(pairId);
  }
  useBookmarkedTurnsStore.getState().forgetThreads(result.deletedThreadIds);
  if (result.backend === "cleared") {
    // Unpin what was cleared, so a never-imported comparison's queued pin cannot block later saves.
    const failed = new Set(result.failedThreadIds);
    const removed = pinnedBefore.filter((id) => !failed.has(id));
    if (removed.length > 0)
      usePinnedChatsStore.getState().setPinned(removed, false);
  }
  return result;
}
