// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState } from "react";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  notifyChatHistoryUpdated,
} from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ThreadRecord } from "../types";
import {
  deleteStoredChatThreads,
  isExpectedBackgroundChatStorageError,
  listStoredChatThreads,
  listStoredChatThreadsWithMessages,
  updateStoredChatThread,
} from "../utils/chat-history-storage";
import { clearBranchHead } from "../utils/branch-head";
import { clearComposerDraft } from "../utils/composer-draft";
import { offerToDeleteKeptSandboxes } from "../utils/offer-kept-sandbox-files";
import { stopChatThread } from "../utils/stop-chat-thread";
import {
  markChatThreadsDeleted,
  removeChatThreadTombstones,
} from "../utils/chat-thread-tombstones";
import { requestPromptQueueStop } from "../utils/prompt-queue-boundary";
import { repairLegacyChatTitles } from "../utils/repair-legacy-chat-titles";

export interface SidebarItem {
  type: "single" | "compare";
  id: string;
  threadIds?: string[];
  title: string;
  createdAt: number;
  updatedAt: number;
  modifiedAt?: number;
  modelIds?: string[];
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
    // Legacy threads may have archived undefined or null; treat missing as false.
    if (Boolean(t.archived) !== archived) {
      continue;
    }
    if (t.pairId) {
      const existing = pairItems.get(t.pairId);
      if (existing) {
        existing.createdAt = Math.max(existing.createdAt, t.createdAt);
        existing.updatedAt = Math.max(existing.updatedAt, lastActivityAt(t));
        if (t.modifiedAt) existing.modifiedAt = Math.max(existing.modifiedAt ?? 0, t.modifiedAt);
        if (t.modelId && !existing.modelIds?.includes(t.modelId)) {
          existing.modelIds = [...(existing.modelIds ?? []), t.modelId];
        }
        existing.threadIds?.push(t.id);
        continue;
      }
      const item: SidebarItem = {
        type: "compare",
        id: t.pairId,
        threadIds: [t.id],
        title: t.title,
        createdAt: t.createdAt,
        updatedAt: lastActivityAt(t),
        ...(t.modifiedAt ? { modifiedAt: t.modifiedAt } : {}),
        ...(t.modelId ? { modelIds: [t.modelId] } : {}),
        projectId: t.projectId ?? null,
      };
      pairItems.set(t.pairId, item);
      items.push(item);
    } else if (!t.pairId) {
      items.push({
        type: "single",
        id: t.id,
        threadIds: [t.id],
        title: t.title,
        createdAt: t.createdAt,
        updatedAt: lastActivityAt(t),
        ...(t.modifiedAt ? { modifiedAt: t.modifiedAt } : {}),
        ...(t.modelId ? { modelIds: [t.modelId] } : {}),
        isFork: Boolean(t.forkedFromThreadId),
        projectId: t.projectId ?? null,
      });
    }
  }

  return items.sort((a, b) => b.updatedAt - a.updatedAt);
}

// Streaming fires the history event per chunk; debounce and discard stale responses.
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
        const threads = await listThreads({
          includeArchived: true,
          projectId: options?.projectId,
        });
        if (cancelled || seq !== requestSeq) return;
        setAllThreads(threads);
        setLoaded(true);
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

    requestSeq += 1;
    void doLoad(requestSeq);
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, load);
    return () => {
      cancelled = true;
      if (pendingTimer !== null) clearTimeout(pendingTimer);
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, load);
    };
  }, [enabled, options?.projectId, requireMessages]);

  // Memoised for identity: new arrays per render make dependent effects loop forever.
  const items = useMemo(() => groupThreads(allThreads ?? []), [allThreads]);
  const archivedItems = useMemo(
    () => groupThreads(allThreads ?? [], true),
    [allThreads],
  );
  const canCompare = useChatRuntimeStore((s) => Boolean(s.params.checkpoint));

  return { items, archivedItems, canCompare, loaded };
}

function cancelIfRunning(threadId: string): void {
  // cancelByThreadId cannot reach background threads; a deleted chat must stop.
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
  const toArchive = threads.filter((t) => !t.archived);
  if (toArchive.length === 0) return 0;

  requestPromptQueueStop(toArchive.map((thread) => thread.id));
  for (const t of toArchive) cancelIfRunning(t.id);

  // allSettled: Promise.all rejects while slower siblings are still writing.
  const writes = await Promise.allSettled(
    toArchive.map((t) =>
      updateStoredChatThread(t.id, { archived: true }, { notify: false }),
    ),
  );
  const failure = writes.find(
    (write): write is PromiseRejectedResult => write.status === "rejected",
  );
  if (failure) {
    notifyChatHistoryUpdated();
    throw failure.reason;
  }

  // Reset only if this action archived the active chat.
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

  requestPromptQueueStop(threadIds);

  for (const id of threadIds) {
    cancelIfRunning(id);
  }

  // clear composer drafts so deleted threads leave no orphan keys.
  for (const id of threadIds) clearComposerDraft(id);

  markChatThreadsDeleted(threadIds);
  notifyChatHistoryUpdated();

  if (activeId !== undefined && items.some((item) => item.id === activeId)) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }

  try {
    const kept = await deleteStoredChatThreads(threadIds, args);
    // retain the head until deletion succeeds so a restored chat reopens on the same branch.
    for (const id of threadIds) clearBranchHead(id);
    // offer recovery for sandbox files that outlive a deleted chat and lose their sidebar entry.
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
