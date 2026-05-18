// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ThreadRecord } from "../types";
import {
  deleteStoredChatThreads,
  listStoredChatThreads,
  listStoredChatThreadsWithMessages,
  updateStoredChatThread,
} from "../utils/chat-history-storage";

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
      const threads = await listStoredChatThreadsWithMessages({
        includeArchived: false,
      });
      if (!cancelled) setAllThreads(threads);
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

  await deleteStoredChatThreads(threadIds);

  if (activeId === item.id) {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    onSelect({ mode: "single", newThreadNonce: crypto.randomUUID() });
  }
}
