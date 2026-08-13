// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch } from "@/features/auth";
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
  // Lowercased title + user messages only (short); searched first.
  userSearchText: string;
  // Lowercased title + every message (incl. tool calls); fallback when user
  // text matches nothing. Prebuilt so filtering never re-lowercases per keystroke.
  searchText: string;
  createdAt: number;
  projectId?: string | null;
}

const THREAD_LIMIT = 200;
const SEARCH_REBUILD_DEBOUNCE_MS = 300;

// Keys whose values are base64 image/audio payloads, not searchable text.
const BINARY_KEY = /b64|base64|^(images?|audio|video)$/i;

// Drop a trailing __MCP_IMAGES__ envelope only when it is the valid JSON image
// array appended by the backend, so legit tool text that merely mentions the
// marker stays searchable. (base64 runs below are scrubbed regardless.)
function stripMcpImageSuffix(value: string): string {
  const marker = "\n__MCP_IMAGES__:";
  const idx = value.lastIndexOf(marker);
  if (idx === -1) return value;
  try {
    const images: unknown = JSON.parse(value.slice(idx + marker.length));
    if (
      Array.isArray(images) &&
      images.length > 0 &&
      images.every(
        (img) =>
          typeof img === "object" &&
          img !== null &&
          typeof (img as Record<string, unknown>).data === "string" &&
          typeof (img as Record<string, unknown>).mimeType === "string",
      )
    ) {
      return value.slice(0, idx);
    }
  } catch {
    // Not a valid envelope; leave the text intact.
  }
  return value;
}

// Readable text from tool args/results, dropping base64 image/audio blobs so
// they never bloat the index (object fields by key, plus data URLs / long
// base64 runs and the "__IMAGES__" suffix inside strings).
function searchableText(value: unknown, depth = 0): string {
  if (typeof value === "string") {
    let text = stripMcpImageSuffix(value);
    const cut = text.indexOf("\n__IMAGES__:");
    if (cut !== -1) text = text.slice(0, cut);
    return text
      .replace(/data:[^;,\s]+;base64,[A-Za-z0-9+/=]+/g, " ")
      .replace(/[A-Za-z0-9+/]{120,}={0,2}/g, " ");
  }
  if (value == null || depth > 4) return "";
  if (Array.isArray(value)) {
    return value.map((v) => searchableText(v, depth + 1)).join(" ");
  }
  if (typeof value === "object") {
    const out: string[] = [];
    for (const [k, v] of Object.entries(value)) {
      if (!BINARY_KEY.test(k)) out.push(searchableText(v, depth + 1));
    }
    return out.join(" ");
  }
  return "";
}

// Pull searchable text from a message: plain text, reasoning/thinking, tool
// calls (name + args + result) and cited sources (title + url).
function extractText(message: MessageRecord): string {
  const content = message.content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as Record<string, unknown>;
    if ((p.type === "text" || p.type === "reasoning") && typeof p.text === "string") {
      parts.push(p.text);
    } else if (p.type === "thinking") {
      const t = typeof p.thinking === "string" ? p.thinking : p.text;
      if (typeof t === "string") parts.push(t);
    } else if (p.type === "tool-call") {
      if (typeof p.toolName === "string") parts.push(p.toolName);
      const args = searchableText(typeof p.argsText === "string" ? p.argsText : p.args);
      if (args) parts.push(args);
      const result = searchableText(p.result);
      if (result) parts.push(result);
    } else if (p.type === "source") {
      for (const v of [p.title, p.url]) if (typeof v === "string") parts.push(v);
    }
  }
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

async function buildIndex(): Promise<ChatSearchItem[]> {
  const active = (
    await listStoredChatThreads({ includeArchived: false })
  ).slice(0, THREAD_LIMIT);

  const itemThreadIds = new Map<
    string,
    {
      item: Omit<ChatSearchItem, "searchText" | "userSearchText">;
      threadIds: string[];
    }
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
  // Fill only the missing ids via the legacy path instead of one request per
  // thread up front.
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

    // Two tiers: user messages (short, searched first) and the full
    // conversation incl. tool calls (fallback when user text matches nothing).
    const userParts: string[] = [item.title];
    const allParts: string[] = [item.title];
    for (const m of merged) {
      const text = extractText(m);
      if (!text) continue;
      allParts.push(text);
      if (m.role === "user") userParts.push(text);
    }
    const userSearchText = userParts.join(" ").toLowerCase();
    const searchText = allParts.join(" ").toLowerCase();
    results.push({ ...item, userSearchText, searchText });
  }

  results.sort((a, b) => b.createdAt - a.createdAt);
  return results;
}

// Last built index, kept across opens so reopening paints the previous rows at
// once and revalidates in place instead of collapsing back to the empty state.
let cachedIndex: ChatSearchItem[] | null = null;
let cachedIndexEpoch = -1;

// The cache outlives the sidebar and a web logout only navigates, so it is scoped to the
// auth session: a second account must never open onto the previous user's chats.
function readCachedIndex(): ChatSearchItem[] | null {
  if (cachedIndexEpoch !== getAuthSessionEpoch()) cachedIndex = null;
  return cachedIndex;
}

function writeCachedIndex(next: ChatSearchItem[] | null): void {
  cachedIndex = next;
  cachedIndexEpoch = getAuthSessionEpoch();
}

// How many rows a completed build left behind, readable during render so the dialog can size
// itself before its opening paint. An index that has not been built yet counts as none: the
// first open of a page load must not reserve a height it may never fill.
export function cachedChatSearchIndexItemCount(): number {
  return readCachedIndex()?.length ?? 0;
}

export function useChatSearchIndex(enabled: boolean): {
  items: ChatSearchItem[];
  loading: boolean;
} {
  const [items, setItems] = useState<ChatSearchItem[]>(() => readCachedIndex() ?? []);
  const [loading, setLoading] = useState(false);
  const requestSeqRef = useRef(0);

  // Opening onto an invalidated cache must not paint the previous rows, so discard them in
  // the opening render rather than in the effect that rebuilds, which runs after the commit.
  const [wasEnabled, setWasEnabled] = useState(enabled);
  if (enabled !== wasEnabled) {
    setWasEnabled(enabled);
    if (enabled && readCachedIndex() === null) {
      if (items.length > 0) setItems([]);
      if (!loading) setLoading(true);
    }
  }

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      // History can change while the dialog is closed, so drop the cache rather
      // than reopening onto rows for chats that no longer exist. Only the cache is
      // touched: clearing state would also re-render for every streaming chunk.
      const invalidate = () => {
        writeCachedIndex(null);
      };
      window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, invalidate);
      return () => window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, invalidate);
    }
    let cancelled = false;
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;
    // Set by a history event and cleared once its rebuild lands, so a close in between
    // can tell that the cached snapshot is known to be out of date.
    let rebuildPending = false;

    const run = () => {
      const seq = ++requestSeqRef.current;
      // A build that straddles a logout describes the account it started under, so its
      // result belongs to that session alone.
      const epoch = getAuthSessionEpoch();
      // Only the first build has nothing to show; later ones refresh silently.
      if (readCachedIndex() === null) setLoading(true);
      buildIndex()
        .then((result) => {
          // Drop out-of-order responses so a slower rebuild can't clobber a fresher one,
          // and never repopulate the cache a close or an invalidation already dropped.
          if (cancelled || seq !== requestSeqRef.current) return;
          if (epoch !== getAuthSessionEpoch()) return;
          writeCachedIndex(result);
          // A build that started before the history event does not satisfy it, so the flag
          // only clears once no rebuild is still queued.
          if (debounceTimer === null) rebuildPending = false;
          setItems(result);
        })
        .catch(() => {
          if (cancelled || seq !== requestSeqRef.current) return;
          setItems(readCachedIndex() ?? []);
        })
        .finally(() => {
          if (cancelled || seq !== requestSeqRef.current) return;
          setLoading(false);
        });
    };

    const scheduleRebuild = () => {
      rebuildPending = true;
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
      // Closing before a history change was rebuilt cancels that rebuild, so the snapshot
      // left behind is known stale and must not survive into the next open. The rendered
      // rows stay put: clearing them here would tear the list down inside the exit animation.
      if (rebuildPending) writeCachedIndex(null);
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, scheduleRebuild);
    };
  }, [enabled]);

  return { items, loading };
}
