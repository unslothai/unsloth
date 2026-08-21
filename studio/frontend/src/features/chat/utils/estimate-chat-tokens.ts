// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import type { MessageRecord } from "../types";

/**
 * Extracts raw text content from message records or ThreadMessages across the active branch.
 */
export function extractMessageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let text = "";
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as { type?: string; text?: unknown };
    if (typeof p.text === "string") {
      text += p.text;
    }
  }
  return text;
}

const ROLE_ORDER: Record<string, number> = { system: 0, user: 1, assistant: 2 };

/**
 * Traverses messages to find the selected branch.
 */
export function orderMessagesByBranch<
  T extends { id: string; createdAt: number | Date; role: string; parentId?: string | null },
>(messages: readonly T[]): T[] {
  const toTime = (v: number | Date): number =>
    typeof v === "number" ? v : v.getTime();

  const sorted = messages.slice().sort((a, b) => {
    const timeA = toTime(a.createdAt);
    const timeB = toTime(b.createdAt);
    if (timeA !== timeB) return timeA - timeB;
    const aOrder = ROLE_ORDER[a.role] ?? 99;
    const bOrder = ROLE_ORDER[b.role] ?? 99;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });

  const byId = new Map<string, T>();
  const parentOf = new Map<string, string | null>();
  let previousId: string | null = null;
  for (const m of sorted) {
    byId.set(m.id, m);
    parentOf.set(m.id, m.parentId ?? previousId);
    previousId = m.id;
  }

  const chain: T[] = [];
  const seen = new Set<string>();
  let cur: string | null = sorted.at(-1)?.id ?? null;
  while (cur != null && !seen.has(cur)) {
    seen.add(cur);
    const record = byId.get(cur);
    if (!record) break;
    chain.push(record);
    cur = parentOf.get(cur) ?? null;
  }
  return chain.reverse();
}

/**
 * Estimate token count from a list of stored message records or ThreadMessages (char length / 4).
 * Returns null if no text is found.
 */
export function estimateMessagesTokenCount(
  messages: readonly (MessageRecord | ThreadMessage)[] | null | undefined,
): number | null {
  if (!messages || messages.length === 0) return null;

  const branch = orderMessagesByBranch(
    messages as readonly {
      id: string;
      createdAt: number | Date;
      role: string;
      parentId?: string | null;
      content: MessageRecord["content"];
    }[],
  );

  let totalChars = 0;
  for (const msg of branch) {
    const text = extractMessageText(msg.content);
    totalChars += text.length;
  }

  if (totalChars === 0) return null;
  return Math.max(1, Math.round(totalChars / 4));
}
