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
          // Drop out-of-order responses so a slower rebuild can't clobber a fresher one.
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
