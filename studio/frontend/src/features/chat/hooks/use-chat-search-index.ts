// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { batchListChatMessages, CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";
import type { MessageRecord } from "../types";
import {
  listStoredChatMessages,
  listStoredChatThreads,
} from "../utils/chat-history-storage";

export interface ChatSearchItem {
  type: "single" | "compare";
  id: string;
  title: string;
  preview: string;
  createdAt: number;
  projectId?: string | null;
}

const THREAD_LIMIT = 200;
const PREVIEW_MAX = 120;
const SEARCH_REBUILD_DEBOUNCE_MS = 300;

function extractText(message: MessageRecord): string {
  const content = message.content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as { type?: string; text?: unknown };
    if (
      (p.type === "text" || p.type === "reasoning") &&
      typeof p.text === "string"
    ) {
      parts.push(p.text);
    }
  }
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max).trimEnd()}…`;
}

async function buildIndex(): Promise<ChatSearchItem[]> {
  const active = (
    await listStoredChatThreads({ includeArchived: false })
  ).slice(0, THREAD_LIMIT);

  const itemThreadIds = new Map<
    string,
    { item: Omit<ChatSearchItem, "preview">; threadIds: string[] }
  >();
  const seenPairs = new Set<string>();

  for (const t of active) {
    if (t.pairId) {
      if (seenPairs.has(t.pairId)) {
        const existing = itemThreadIds.get(t.pairId);
        if (existing) existing.threadIds.push(t.id);
        continue;
      }
      seenPairs.add(t.pairId);
      itemThreadIds.set(t.pairId, {
        item: {
          type: "compare",
          id: t.pairId,
          title: t.title,
          createdAt: t.createdAt,
          projectId: t.projectId ?? null,
        },
        threadIds: [t.id],
      });
    } else {
      itemThreadIds.set(t.id, {
        item: {
          type: "single",
          id: t.id,
          title: t.title,
          createdAt: t.createdAt,
          projectId: t.projectId ?? null,
        },
        threadIds: [t.id],
      });
    }
  }

  const allThreadIds = Array.from(itemThreadIds.values()).flatMap(
    (e) => e.threadIds,
  );
  let messagesByThread = await batchListChatMessages(allThreadIds).catch(
    () => new Map<string, MessageRecord[]>(),
  );

  // Legacy-only chats can exist before server-side history import finishes.
  // Fill just the missing ids from the legacy-aware path instead of issuing
  // one request per thread up front.
  const missingThreadIds = allThreadIds.filter(
    (threadId) => !messagesByThread.has(threadId),
  );
  if (missingThreadIds.length > 0) {
    const legacyEntries = await Promise.all(
      missingThreadIds.map(async (threadId) => [
        threadId,
        await listStoredChatMessages(threadId).catch(() => []),
      ] as const),
    );
    messagesByThread = new Map(messagesByThread);
    for (const [threadId, messages] of legacyEntries) {
      messagesByThread.set(threadId, messages);
    }
  }

  const results: ChatSearchItem[] = [];
  for (const { item, threadIds } of itemThreadIds.values()) {
    const merged: MessageRecord[] = [];
    for (const tid of threadIds) {
      const arr = messagesByThread.get(tid);
      if (arr) merged.push(...arr);
    }
    if (merged.length === 0) {
      continue;
    }
    merged.sort((a, b) => b.createdAt - a.createdAt);

    let preview = "";
    for (const m of merged) {
      const text = extractText(m);
      if (text) {
        preview = truncate(text, PREVIEW_MAX);
        break;
      }
    }
    results.push({ ...item, preview });
  }

  results.sort((a, b) => b.createdAt - a.createdAt);
  return results;
}

export function useChatSearchIndex(enabled: boolean): {
  items: ChatSearchItem[];
  loading: boolean;
} {
  const [items, setItems] = useState<ChatSearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const requestSeqRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      // Clear stale results so the next open doesn't flash old items.
      setItems([]);
      setLoading(false);
      return;
    }
    let cancelled = false;
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;

    const run = () => {
      const seq = ++requestSeqRef.current;
      setLoading(true);
      buildIndex()
        .then((result) => {
          // Drop out-of-order responses so a slower rebuild can't clobber
          // a fresher one.
          if (cancelled || seq !== requestSeqRef.current) return;
          setItems(result);
        })
        .catch(() => {
          if (cancelled || seq !== requestSeqRef.current) return;
          setItems([]);
        })
        .finally(() => {
          if (cancelled || seq !== requestSeqRef.current) return;
          setLoading(false);
        });
    };

    const scheduleRebuild = () => {
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        debounceTimer = null;
        if (!cancelled) run();
      }, SEARCH_REBUILD_DEBOUNCE_MS);
    };

    run();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, scheduleRebuild);
    return () => {
      cancelled = true;
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, scheduleRebuild);
    };
  }, [enabled]);

  return { items, loading };
}
