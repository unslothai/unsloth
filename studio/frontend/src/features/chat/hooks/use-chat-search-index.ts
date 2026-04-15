// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { db } from "../db";
import type { MessageRecord, ThreadRecord } from "../types";

export interface ChatSearchItem {
  type: "single" | "compare";
  id: string;
  title: string;
  preview: string;
  createdAt: number;
}

const THREAD_LIMIT = 200;
const PREVIEW_MAX = 120;

function extractText(message: MessageRecord): string {
  const content = message.content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as { type?: string; text?: unknown };
    if ((p.type === "text" || p.type === "reasoning") && typeof p.text === "string") {
      parts.push(p.text);
    }
  }
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max).trimEnd() + "…";
}

async function buildIndex(): Promise<ChatSearchItem[]> {
  // Fetch all threads newest-first, filter archived in JS, then take top N.
  // `archived` is a boolean which Dexie does not index reliably, so we filter
  // after the sort instead of using `.where("archived")`.
  const all = (await db.threads
    .orderBy("createdAt")
    .reverse()
    .toArray()) as ThreadRecord[];
  const active = all.filter((t) => !t.archived).slice(0, THREAD_LIMIT);

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
        },
        threadIds: [t.id],
      });
    }
  }

  // One query for all messages across all relevant threads, then group by
  // threadId in memory. Avoids N sequential awaits.
  const allThreadIds = Array.from(itemThreadIds.values()).flatMap(
    (e) => e.threadIds,
  );
  const messages = (await db.messages
    .where("threadId")
    .anyOf(allThreadIds)
    .toArray()) as MessageRecord[];

  const byThreadId = new Map<string, MessageRecord[]>();
  for (const m of messages) {
    const arr = byThreadId.get(m.threadId);
    if (arr) arr.push(m);
    else byThreadId.set(m.threadId, [m]);
  }

  const results: ChatSearchItem[] = [];
  for (const { item, threadIds } of itemThreadIds.values()) {
    const merged: MessageRecord[] = [];
    for (const tid of threadIds) {
      const arr = byThreadId.get(tid);
      if (arr) merged.push(...arr);
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

  useEffect(() => {
    if (!enabled) {
      // Clear stale results so the next open doesn't flash old items.
      setItems([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    buildIndex()
      .then((result) => {
        if (!cancelled) setItems(result);
      })
      .catch(() => {
        if (!cancelled) setItems([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  return { items, loading };
}
