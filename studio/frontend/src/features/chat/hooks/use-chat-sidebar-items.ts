// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { db, useLiveQuery } from "../db";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ThreadRecord } from "../types";

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
  const allThreads = useLiveQuery(async () => {
    const threadIdsWithMessage = new Set(
      (await db.messages.orderBy("threadId").uniqueKeys()) as string[],
    );
    const rows = await db.threads.orderBy("createdAt").reverse().toArray();
    return rows.filter((t) => !t.archived && threadIdsWithMessage.has(t.id));
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
      : (await db.threads.where("pairId").equals(item.id).toArray()).map(
          (t) => t.id,
        );

  // Stop any in-flight streams before deleting, so the model doesn't keep
  // generating against a thread that no longer exists.
  for (const id of threadIds) cancelIfRunning(id);

  await db.transaction("rw", db.threads, db.messages, async () => {
    for (const id of threadIds) {
      await db.messages.where("threadId").equals(id).delete();
      await db.threads.delete(id);
    }
  });

  if (activeId === item.id) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }
}
