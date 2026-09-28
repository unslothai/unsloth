// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type SidebarItem, listStoredChatMessagesMany } from "@/features/chat";
import { listAllDocuments } from "@/features/rag";
import { useEffect, useMemo, useState } from "react";
import { type ChatContents, summarizeChatMessages } from "./model";

// Read once per chat version; a new message changes `updatedAt` and so the key.
const cache = new Map<string, ChatContents>();
const BATCH = 20;

function cacheKey(chat: SidebarItem): string {
  return `${chat.id}:${chat.updatedAt}`;
}

function threadIdsOf(chat: SidebarItem): string[] {
  return chat.threadIds?.length ? chat.threadIds : [chat.id];
}

/** A compare chat counts its fullest pane: the panes share the user's messages. */
function combine(summaries: ChatContents[]): ChatContents {
  return summaries.reduce(
    (best, next) => (next.messages > best.messages ? next : best),
    { messages: 0, images: 0, html: 0 },
  );
}

/** Contents of the listed chats, read in small batches as they come into the list. */
export function useChatContents(chats: readonly SidebarItem[]): ReadonlyMap<string, ChatContents> {
  const [version, setVersion] = useState(0);
  const missingKey = chats
    .filter((chat) => !cache.has(cacheKey(chat)))
    .map(cacheKey)
    .join(",");
  // biome-ignore lint/correctness/useExhaustiveDependencies: the missing keys are the cue to read
  useEffect(() => {
    const missing = chats.filter((chat) => !cache.has(cacheKey(chat)));
    if (missing.length === 0) return;
    let cancelled = false;
    void (async () => {
      for (let i = 0; i < missing.length && !cancelled; i += BATCH) {
        const batch = missing.slice(i, i + BATCH);
        const byThread = await listStoredChatMessagesMany(batch.flatMap(threadIdsOf)).catch(
          () => null,
        );
        if (!byThread || cancelled) return;
        for (const chat of batch) {
          cache.set(
            cacheKey(chat),
            combine(threadIdsOf(chat).map((id) => summarizeChatMessages(byThread.get(id) ?? []))),
          );
        }
        setVersion((v) => v + 1);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [missingKey]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: version marks new cache entries
  return useMemo(() => {
    const out = new Map<string, ChatContents>();
    for (const chat of chats) {
      const entry = cache.get(cacheKey(chat));
      if (entry) out.set(chat.id, entry);
    }
    return out;
  }, [chats, version]);
}

/** Sources per project; null while loading or when document search is unavailable. */
export function useProjectSourceCounts(enabled: boolean): ReadonlyMap<string, number> | null {
  const [counts, setCounts] = useState<Map<string, number> | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    listAllDocuments()
      .then((documents) => {
        if (cancelled) return;
        const next = new Map<string, number>();
        for (const document of documents) {
          if (!document.projectId || document.threadId) continue;
          next.set(document.projectId, (next.get(document.projectId) ?? 0) + 1);
        }
        setCounts(next);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [enabled]);
  return counts;
}
