// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  deleteChatThreads,
  listChatMessages,
  listChatThreads,
} from "../api/chat-api";
import { db } from "../db";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ThreadRecord } from "../types";
import {
  isChatThreadDeleted,
  markChatThreadDeleted,
} from "../utils/chat-thread-tombstones";

export interface SidebarItem {
  type: "single" | "compare";
  id: string;
  title: string;
  createdAt: number;
}

export function groupThreads(threads: ThreadRecord[]): SidebarItem[] {
  const items: SidebarItem[] = [];
  const seenPairs = new Set<string>();

  for (const t of threads) {
    if (t.archived) {
      continue;
    }
    if (t.pairId) {
      if (seenPairs.has(t.pairId)) {
        continue;
      }
      seenPairs.add(t.pairId);
      items.push({
        type: "compare",
        id: t.pairId,
        title: t.title,
        createdAt: t.createdAt,
      });
    } else if (!t.pairId) {
      items.push({
        type: "single",
        id: t.id,
        title: t.title,
        createdAt: t.createdAt,
      });
    }
  }

  return items.sort((a, b) => b.createdAt - a.createdAt);
}

export function useChatSidebarItems() {
  const [allThreads, setAllThreads] = useState<ThreadRecord[]>([]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const [backendThreads, legacyThreadIdsWithMessage, legacyThreads] =
        await Promise.all([
          listChatThreads({ includeArchived: false }).catch(() => []),
          db.messages.orderBy("threadId").uniqueKeys() as Promise<string[]>,
          db.threads.orderBy("createdAt").reverse().toArray(),
        ]);
      const backendIds = new Set(backendThreads.map((t) => t.id));
      const legacyIdsWithMessage = new Set(legacyThreadIdsWithMessage);
      const merged = new Map<string, ThreadRecord>();
      for (const thread of legacyThreads) {
        if (
          thread.archived ||
          isChatThreadDeleted(thread.id) ||
          backendIds.has(thread.id) ||
          !legacyIdsWithMessage.has(thread.id)
        ) {
          continue;
        }
        merged.set(thread.id, thread);
      }
      for (const thread of backendThreads) {
        if (isChatThreadDeleted(thread.id)) continue;
        const messages = await listChatMessages(thread.id).catch(() => []);
        if (messages.length > 0) merged.set(thread.id, thread);
      }
      if (!cancelled) setAllThreads(Array.from(merged.values()));
    }
    void load();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, load);
    return () => {
      cancelled = true;
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, load);
    };
  }, []);

  const items = groupThreads(allThreads ?? []);
  const canCompare = useChatRuntimeStore((s) => Boolean(s.params.checkpoint));

  return { items, canCompare };
}

function cancelIfRunning(threadId: string): void {
  const { runningByThreadId, cancelByThreadId } =
    useChatRuntimeStore.getState();
  if (!runningByThreadId[threadId]) return;
  cancelByThreadId[threadId]?.();
}

export async function deleteChatItem(
  item: SidebarItem,
  activeId: string | undefined,
  onSelect: (view: { mode: "single"; newThreadNonce: string }) => void,
) {
  const threadIds: string[] =
    item.type === "single"
      ? [item.id]
      : [
          ...(await listChatThreads({ pairId: item.id }).catch(() => [])),
          ...(await db.threads.where("pairId").equals(item.id).toArray()),
        ].map((t) => t.id);

  // Stop any in-flight streams before deleting, so the model doesn't keep
  // generating against a thread that no longer exists.
  for (const id of threadIds) cancelIfRunning(id);
  for (const id of threadIds) markChatThreadDeleted(id);

  await deleteChatThreads(threadIds);

  if (activeId === item.id) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }
}
