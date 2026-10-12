// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SidebarItem,
  countStoredChatMessages,
  listStoredChatMessagesMany,
} from "@/features/chat";
import { useEffect, useMemo, useState } from "react";
import { type ChatContents, summarizeChatMessages } from "./model";

// Keyed by chat version (`updatedAt`). Module-level to survive tab switches; capped.
const cache = new Map<string, ChatContents>();
const MAX_CACHED = 5000;
// Keys in flight or failed, so re-renders never re-request. A failed key retries on a new version.
const pending = new Set<string>();
const failed = new Set<string>();
const listeners = new Set<() => void>();
// Counts are tiny; the older-server fallback reads whole messages.
const BATCH = 100;
const FALLBACK_BATCH = 20;

function cacheKey(chat: SidebarItem): string {
  return `${chat.id}:${chat.updatedAt}`;
}

function threadIdsOf(chat: SidebarItem): string[] {
  return chat.threadIds?.length ? chat.threadIds : [chat.id];
}

function remember(key: string, value: ChatContents): void {
  cache.set(key, value);
  if (cache.size > MAX_CACHED) cache.delete(cache.keys().next().value as string);
}

function needsRead(chat: SidebarItem): boolean {
  const key = cacheKey(chat);
  return !cache.has(key) && !pending.has(key) && !failed.has(key);
}

/** A compare chat counts its fullest pane: the panes share the user's messages. */
function combine(counts: number[]): ChatContents {
  return { messages: counts.reduce((best, next) => Math.max(best, next), 0) };
}

async function readCounts(chats: SidebarItem[]): Promise<(threadId: string) => number> {
  const ids = chats.flatMap(threadIdsOf);
  const counts = await countStoredChatMessages(ids);
  if (counts) return (id) => counts.get(id) ?? 0;
  const byThread = new Map<string, number>();
  for (let i = 0; i < ids.length; i += FALLBACK_BATCH) {
    const read = await listStoredChatMessagesMany(ids.slice(i, i + FALLBACK_BATCH));
    for (const [id, messages] of read) byThread.set(id, summarizeChatMessages(messages).messages);
  }
  return (id) => byThread.get(id) ?? 0;
}

async function readContents(chats: SidebarItem[]): Promise<void> {
  const keys = chats.map(cacheKey);
  for (const key of keys) pending.add(key);
  try {
    for (let i = 0; i < chats.length; i += BATCH) {
      const batch = chats.slice(i, i + BATCH);
      try {
        const count = await readCounts(batch);
        for (const chat of batch) remember(cacheKey(chat), combine(threadIdsOf(chat).map(count)));
      } catch {
        for (const chat of batch) failed.add(cacheKey(chat));
      }
      for (const listener of listeners) listener();
    }
  } finally {
    for (const key of keys) pending.delete(key);
  }
}

export function useChatContents(chats: readonly SidebarItem[]): ReadonlyMap<string, ChatContents> {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const listener = () => setVersion((v) => v + 1);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);
  const missingKey = chats.filter(needsRead).map(cacheKey).join(",");
  // biome-ignore lint/correctness/useExhaustiveDependencies: the missing keys are the cue to read
  useEffect(() => {
    const missing = chats.filter(needsRead);
    // Not cancelled: a read in flight still fills the shared cache.
    if (missing.length > 0) void readContents(missing);
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
