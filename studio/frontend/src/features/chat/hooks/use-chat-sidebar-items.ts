// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  notifyChatHistoryUpdated,
} from "../api/chat-api";
import { useChatArtifactsStore } from "../artifacts/store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ThreadRecord } from "../types";
import {
  deleteStoredChatThreads,
  isExpectedBackgroundChatStorageError,
  listStoredChatThreads,
  listStoredChatThreadsWithMessages,
  updateStoredChatThread,
} from "../utils/chat-history-storage";
import { clearComposerDraft } from "../utils/composer-draft";
import { offerToDeleteKeptSandboxes } from "../utils/offer-kept-sandbox-files";
import { stopChatThread } from "../utils/stop-chat-thread";
import {
  markChatThreadsDeleted,
  removeChatThreadTombstones,
} from "../utils/chat-thread-tombstones";
import { requestPromptQueueStop } from "../utils/prompt-queue-boundary";
import { repairLegacyChatTitles } from "../utils/repair-legacy-chat-titles";
import {
  groupThreads,
  sameSidebarThreads,
  useSidebarThreadGroups,
  type SidebarItem,
} from "./sidebar-thread-groups";

// Re-exported, not moved: every caller in the tree imports these from here.
export { groupThreads } from "./sidebar-thread-groups";
export type { SidebarItem } from "./sidebar-thread-groups";

// Streaming fires CHAT_HISTORY_UPDATED_EVENT per chunk. Debounce so each quiet
// window produces at most one O(N) fetch; requestSeq discards stale responses.
const SIDEBAR_REFRESH_DEBOUNCE_MS = 300;

export function useChatSidebarItems(options?: {
  projectId?: string | null;
  enabled?: boolean;
  requireMessages?: boolean;
}) {
  const [allThreads, setAllThreads] = useState<ThreadRecord[]>([]);
  const [loaded, setLoaded] = useState(false);
  const enabled = options?.enabled ?? true;
  const requireMessages = options?.requireMessages ?? true;

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let cancelled = false;
    let pendingTimer: ReturnType<typeof setTimeout> | null = null;
    let requestSeq = 0;

    async function doLoad(seq: number) {
      try {
        const listThreads = requireMessages
          ? listStoredChatThreadsWithMessages
          : listStoredChatThreads;
        // includeArchived: archived threads are filtered out of Recents by
        // groupThreads, but the hook still needs them for archivedItems.
        const threads = await listThreads({
          includeArchived: true,
          projectId: options?.projectId,
        });
        // Discard the response if a newer request was scheduled while we
        // were in flight, or if the effect was torn down.
        if (cancelled || seq !== requestSeq) return;
        // Keep the previous array when the refetch changed nothing the sidebar
        // reads. Every CHAT_HISTORY_UPDATED_EVENT refetches the WHOLE list and
        // streaming fires that event per chunk, so without this a quiet refresh
        // handed React a brand new array, re-rendered the rail and regrouped
        // the entire history for no visible difference. Object.is on the state
        // makes an unchanged refresh a genuine no-op instead.
        setAllThreads((previous) =>
          sameSidebarThreads(previous, threads) ? previous : threads,
        );
        setLoaded(true);
        // Pre-cut legacy titles cannot grow with the sidebar. The repair reads
        // its own messages, as late as it can, not this list's.
        void repairLegacyChatTitles(threads).catch(() => undefined);
      } catch (error) {
        if (isExpectedBackgroundChatStorageError(error)) {
          return;
        }
        if (!cancelled) throw error;
      }
    }

    function load() {
      if (pendingTimer !== null) clearTimeout(pendingTimer);
      pendingTimer = setTimeout(() => {
        pendingTimer = null;
        requestSeq += 1;
        void doLoad(requestSeq);
      }, SIDEBAR_REFRESH_DEBOUNCE_MS);
    }

    // Initial load fires immediately (no debounce) so the sidebar isn't
    // blank for 300ms on mount.
    requestSeq += 1;
    void doLoad(requestSeq);
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, load);
    return () => {
      cancelled = true;
      if (pendingTimer !== null) clearTimeout(pendingTimer);
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, load);
    };
  }, [enabled, options?.projectId, requireMessages]);

  const { items, archivedItems } = useSidebarThreadGroups(allThreads);
  const canCompare = useChatRuntimeStore((s) => Boolean(s.params.checkpoint));

  return { items, archivedItems, canCompare, loaded };
}

function cancelIfRunning(threadId: string): void {
  // Reaches a background thread, which cancelByThreadId cannot: a deleted chat must stop,
  // or the run keeps writing to a conversation that is gone.
  stopChatThread(threadId);
}

export async function renameChatItem(
  item: SidebarItem,
  nextTitle: string,
): Promise<void> {
  const trimmed = nextTitle.trim();
  if (!trimmed || trimmed === item.title) return;

  if (item.type === "single") {
    await updateStoredChatThread(item.id, { title: trimmed });
    return;
  }

  const threads = await listStoredChatThreads({
    pairId: item.id,
    includeArchived: true,
  });
  const threadIds = Array.from(new Set(threads.map((thread) => thread.id)));
  await Promise.all(
    threadIds.map((id) => updateStoredChatThread(id, { title: trimmed })),
  );
}

async function collectItemThreadIds(
  items: SidebarItem[],
  args: { includeArchived?: boolean } = {},
): Promise<string[]> {
  const ids = new Set<string>();
  for (const item of items) {
    if (item.type === "single") {
      ids.add(item.id);
      continue;
    }
    const pair = await listStoredChatThreads({ pairId: item.id, ...args });
    for (const thread of pair) ids.add(thread.id);
  }
  return Array.from(ids);
}

export async function archiveChatItems(
  items: SidebarItem[],
  activeId: string | undefined,
  onSelect: (view: { mode: "single"; newThreadNonce: string }) => void,
): Promise<void> {
  const threadIds = await collectItemThreadIds(items, {
    includeArchived: true,
  });

  requestPromptQueueStop(threadIds);

  for (const id of threadIds) {
    cancelIfRunning(id);
  }

  await Promise.all(
    threadIds.map((id) => updateStoredChatThread(id, { archived: true })),
  );

  if (activeId !== undefined && items.some((item) => item.id === activeId)) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }

  notifyChatHistoryUpdated();
}

export async function archiveChatItem(
  item: SidebarItem,
  activeId: string | undefined,
  onSelect: (view: { mode: "single"; newThreadNonce: string }) => void,
): Promise<void> {
  return archiveChatItems([item], activeId, onSelect);
}

export async function archiveAllChatItems(
  activeId?: string,
  onSelect?: (view: { mode: "single"; newThreadNonce: string }) => void,
): Promise<number> {
  const threads = await listStoredChatThreads({ includeArchived: true });
  // Boolean() mirrors groupThreads: legacy records may have archived
  // undefined/null, which must count as "not archived".
  const toArchive = threads.filter((t) => !t.archived);
  if (toArchive.length === 0) return 0;

  requestPromptQueueStop(toArchive.map((thread) => thread.id));
  for (const t of toArchive) cancelIfRunning(t.id);

  await Promise.all(
    toArchive.map((t) => updateStoredChatThread(t.id, { archived: true })),
  );

  // Reset only when this action archived the active single thread or compare
  // pair. An already-archived chat opened from the archive is not in
  // toArchive and must stay open.
  const archivedActive =
    activeId !== undefined &&
    toArchive.some(
      (thread) => thread.id === activeId || thread.pairId === activeId,
    );
  if (archivedActive) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect?.({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }

  notifyChatHistoryUpdated();
  // Report sidebar items, not raw threads: a compare pair reads as one chat.
  return groupThreads(toArchive).length;
}

export async function unarchiveChatItem(item: SidebarItem): Promise<void> {
  const threadIds: string[] =
    item.type === "single"
      ? [item.id]
      : (
          await listStoredChatThreads({
            pairId: item.id,
            includeArchived: true,
          })
        ).map((t) => t.id);

  await Promise.all(
    threadIds.map((id) => updateStoredChatThread(id, { archived: false })),
  );

  notifyChatHistoryUpdated();
}

export async function deleteChatItems(
  items: SidebarItem[],
  activeId: string | undefined,
  onSelect: (view: { mode: "single"; newThreadNonce: string }) => void,
  args: { deleteFiles?: boolean } = {},
) {
  const threadIds = await collectItemThreadIds(items);

  // Stop queued prompts and in-flight streams before deleting.
  requestPromptQueueStop(threadIds);

  for (const id of threadIds) {
    cancelIfRunning(id);
  }

  // Drop saved composer drafts so deleted threads leave no orphan keys.
  for (const id of threadIds) clearComposerDraft(id);

  const artifactStore = useChatArtifactsStore.getState();
  for (const id of threadIds) artifactStore.clearArtifactsForThread(id);
  artifactStore.clearOrphanedArtifacts();

  // Optimistic tombstone: hide immediately; roll back on backend error.
  markChatThreadsDeleted(threadIds);
  notifyChatHistoryUpdated();

  if (activeId !== undefined && items.some((item) => item.id === activeId)) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }

  try {
    const kept = await deleteStoredChatThreads(threadIds, args);
    // Whether or not deletion was asked for: a sandbox that could not be
    // removed leaves files with no card left to reach them from, and the chat
    // is already gone, so this offer is the only notice and the only retry.
    offerToDeleteKeptSandboxes(kept);
  } catch (error) {
    removeChatThreadTombstones(threadIds);
    notifyChatHistoryUpdated();
    throw error;
  }
}

export async function deleteChatItem(
  item: SidebarItem,
  activeId: string | undefined,
  onSelect: (view: { mode: "single"; newThreadNonce: string }) => void,
  args: { deleteFiles?: boolean } = {},
) {
  return deleteChatItems([item], activeId, onSelect, args);
}
