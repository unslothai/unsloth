// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

type ParentChainMessage = {
  id: string;
  parentId?: string | null;
  createdAt?: number;
  role?: string;
};

const ROLE_ORDER: Record<string, number> = {
  system: 0,
  user: 1,
  assistant: 2,
};

export function sortChatMessages<T extends ParentChainMessage>(
  messages: T[],
): T[] {
  return [...messages].sort((a, b) => {
    if ((a.createdAt ?? 0) !== (b.createdAt ?? 0)) {
      return (a.createdAt ?? 0) - (b.createdAt ?? 0);
    }
    const aOrder = ROLE_ORDER[a.role ?? ""] ?? 99;
    const bOrder = ROLE_ORDER[b.role ?? ""] ?? 99;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
}

/**
 * Order messages for tree import: walk the newest-child chain first, then
 * append abandoned sibling branches. Matches export/import expectations for
 * regeneration history (see #7732).
 */
export function orderByParentChain<T extends ParentChainMessage>(
  messages: T[],
  options: { includeSiblings?: boolean } = {},
): T[] {
  const { includeSiblings = true } = options;
  const byId = new Map<string, T>(messages.map((m) => [m.id, m]));
  const childrenOf = new Map<string | null, T[]>();
  for (const m of messages) {
    const pid = m.parentId ?? null;
    if (!childrenOf.has(pid)) childrenOf.set(pid, []);
    childrenOf.get(pid)!.push(m);
  }

  const result: T[] = [];
  let cur: string | null = null;
  while (childrenOf.has(cur)) {
    const children = childrenOf.get(cur)!;
    if (children.length === 0) break;
    const next = children.reduce((a, b) =>
      (a.createdAt ?? 0) >= (b.createdAt ?? 0) ? a : b,
    );
    result.push(next);
    cur = next.id;
    byId.delete(next.id);
  }

  if (includeSiblings) {
    for (const [, m] of byId) result.push(m);
  }
  return result;
}

/**
 * Regeneration siblings must share the prompting user as parent. Legacy rows
 * with null parentId were previously chained under the prior assistant, which
 * makes resetHead drop middle branches and leaves only 1/2 in the picker.
 */
export function repairAssistantParentIds<T extends ParentChainMessage>(
  messages: T[],
): T[] {
  const sorted = sortChatMessages(messages);
  if (!sorted.some((m) => m.parentId != null)) {
    return sorted;
  }

  const users = sorted.filter((m) => m.role === "user");
  return sorted.map((message) => {
    if (message.role !== "assistant" || message.parentId != null) {
      return message;
    }
    let parent: T | undefined;
    for (const user of users) {
      if ((user.createdAt ?? 0) <= (message.createdAt ?? 0)) {
        parent = user;
      } else {
        break;
      }
    }
    if (!parent) {
      return message;
    }
    return { ...message, parentId: parent.id };
  });
}

/** Leaf of the newest-child chain; used as MessageRepository headId on import. */
export function resolveHeadMessageId(
  messages: Array<Pick<ParentChainMessage, "id" | "parentId" | "createdAt">>,
): string | null {
  const ordered = orderByParentChain(messages, { includeSiblings: false });
  return ordered.at(-1)?.id ?? null;
}

export function prepareBranchedMessagesForImport(
  messages: MessageRecord[],
): MessageRecord[] {
  const repaired = repairAssistantParentIds(messages);
  return orderByParentChain(repaired, { includeSiblings: true });
}
