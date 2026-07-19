// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  notifyChatHistoryUpdated,
} from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useChatArtifactsStore } from "../artifacts/store";
import type { ThreadRecord } from "../types";
import {
  deleteStoredChatThreads,
  isExpectedBackgroundChatStorageError,
  listStoredChatThreads,
  listStoredChatThreadsWithMessages,
  updateStoredChatThread,
} from "../utils/chat-history-storage";
import { clearComposerDraft } from "../utils/composer-draft";
import {
  markChatThreadsDeleted,
  removeChatThreadTombstones,
} from "../utils/chat-thread-tombstones";

export interface SidebarItem {
  type: "single" | "compare";
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  isFork?: boolean;
  projectId?: string | null;
}

function lastActivityAt(thread: ThreadRecord): number {
  return thread.updatedAt ?? thread.createdAt;
}

export function groupThreads(
  threads: ThreadRecord[],
  archived = false,
): SidebarItem[] {
  const items: SidebarItem[] = [];
  const pairItems = new Map<string, SidebarItem>();

  for (const t of threads) {
    // Coerce archived to a boolean before comparing. Legacy threads (from the
    // older browser-only Unsloth, or any record predating the archived field)
    // can have archived === undefined or null; a raw `!== archived` comparison
    // would drop those from BOTH the Recents (archived=false) and Archived
    // (archived=true) lists, hiding existing chats. Treat missing as false.
    if (Boolean(t.archived) !== archived) {
      continue;
    }
    if (t.pairId) {
      const existing = pairItems.get(t.pairId);
      if (existing) {
        existing.updatedAt = Math.max(existing.updatedAt, lastActivityAt(t));
        continue;
      }
      const item: SidebarItem = {
        type: "compare",
        id: t.pairId,
        title: t.title,
        createdAt: t.createdAt,
        updatedAt: lastActivityAt(t),
        projectId: t.projectId ?? null,
      };
      pairItems.set(t.pairId, item);
      items.push(item);
    } else if (!t.pairId) {
      items.push({
        type: "single",
        id: t.id,
        title: t.title,
        createdAt: t.createdAt,
        updatedAt: lastActivityAt(t),
        isFork: Boolean(t.forkedFromThreadId),
        projectId: t.projectId ?? null,
      });
    }
  }

  return items.sort((a, b) => b.updatedAt - a.updatedAt);
}

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
        setAllThreads(threads);
        setLoaded(true);
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

  const items = groupThreads(allThreads ?? []);
  const archivedItems = groupThreads(allThreads ?? [], true);
  const canCompare = useChatRuntimeStore((s) => Boolean(s.params.checkpoint));

  return { items, archivedItems, canCompare, loaded };
}

function cancelIfRunning(threadId: string): void {
  const { runningByThreadId, cancelByThreadId } =
    useChatRuntimeStore.getState();
  if (!runningByThreadId[threadId]) return;
  cancelByThreadId[threadId]?.();
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

export async function archiveChatItem(
  item: SidebarItem,
  activeId: string | undefined,
  onSelect: (view: { mode: "single"; newThreadNonce: string }) => void,
): Promise<void> {
  const threadIds: string[] =
    item.type === "single"
      ? [item.id]
      : (
          await listStoredChatThreads({
            pairId: item.id,
            includeArchived: true,
          })
        ).map((t) => t.id);

  for (const id of threadIds) cancelIfRunning(id);

  await Promise.all(
    threadIds.map((id) => updateStoredChatThread(id, { archived: true })),
  );

  if (activeId === item.id) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }

  notifyChatHistoryUpdated();
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

export async function deleteChatItem(
  item: SidebarItem,
  activeId: string | undefined,
  onSelect: (view: { mode: "single"; newThreadNonce: string }) => void,
) {
  const threadIds: string[] =
    item.type === "single"
      ? [item.id]
      : (await listStoredChatThreads({ pairId: item.id })).map((t) => t.id);

  // Stop any in-flight streams before deleting, so the model doesn't keep
  // generating against a thread that no longer exists.
  for (const id of threadIds) cancelIfRunning(id);

  // Drop saved composer drafts so deleted threads leave no orphan keys.
  for (const id of threadIds) clearComposerDraft(id);

  const artifactStore = useChatArtifactsStore.getState();
  for (const id of threadIds) artifactStore.clearArtifactsForThread(id);
  artifactStore.clearOrphanedArtifacts();

  // Optimistic tombstone: hide immediately; roll back on backend error.
  markChatThreadsDeleted(threadIds);
  notifyChatHistoryUpdated();

  if (activeId === item.id) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }

  try {
    await deleteStoredChatThreads(threadIds);
  } catch (error) {
    removeChatThreadTombstones(threadIds);
    notifyChatHistoryUpdated();
    throw error;
  }
}
