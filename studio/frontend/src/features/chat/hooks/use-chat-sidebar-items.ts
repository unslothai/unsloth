// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  deleteChatThreads,
  listChatMessages,
  listChatThreads,
  updateChatThread,
} from "../api/chat-api";
import { db } from "../db";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ThreadRecord } from "../types";
import {
  isChatThreadDeleted,
  markChatThreadsDeleted,
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

export async function renameChatItem(
  item: SidebarItem,
  nextTitle: string,
): Promise<void> {
  const trimmed = nextTitle.trim();
  if (!trimmed || trimmed === item.title) return;

  if (item.type === "single") {
    await Promise.allSettled([
      updateChatThread(item.id, { title: trimmed }),
      db.threads.update(item.id, { title: trimmed }),
    ]);
    return;
  }

  const [backendThreads, legacyThreads] = await Promise.all([
    listChatThreads({ pairId: item.id, includeArchived: true }).catch(
      () => [],
    ),
    db.threads.where("pairId").equals(item.id).toArray(),
  ]);
  const threadIds = Array.from(
    new Set([...backendThreads, ...legacyThreads].map((thread) => thread.id)),
  );
  await db.transaction("rw", db.threads, async () => {
    for (const id of threadIds) {
      await db.threads.update(id, { title: trimmed });
    }
  });
  await Promise.allSettled(
    threadIds.map((id) => updateChatThread(id, { title: trimmed })),
  );
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
  markChatThreadsDeleted(threadIds);

  await deleteChatThreads(threadIds);
  await db.transaction("rw", db.threads, db.messages, async () => {
    await db.messages.where("threadId").anyOf(threadIds).delete();
    await db.threads.bulkDelete(threadIds);
  });

  if (activeId === item.id) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }
}
