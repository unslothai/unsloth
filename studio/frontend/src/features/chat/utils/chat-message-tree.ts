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
 * Legacy rows may still have null parentId on user turns. Infer those from the
 * prior message before branch ordering so mixed threads keep their chain.
 */
export function inferSequentialParentIds<T extends ParentChainMessage>(
  messages: T[],
): T[] {
  const sorted = sortChatMessages(messages);
  let previousId: string | null = null;
  return sorted.map((message) => {
    if (message.parentId != null) {
      previousId = message.id;
      return message;
    }
    if (message.role === "assistant") {
      return message;
    }
    const inferred = { ...message, parentId: previousId };
    previousId = message.id;
    return inferred;
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
    const children: T[] = childrenOf.get(cur)!;
    if (children.length === 0) break;
    const next: T = children.reduce((a: T, b: T) =>
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

  const parentForOrphanAssistant = (index: number): string | null => {
    for (let i = index - 1; i >= 0; i--) {
      const prev = sorted[i];
      if (prev.role === "user") {
        if (prev.parentId == null) {
          return prev.id;
        }
        continue;
      }
      if (prev.role === "assistant" && prev.parentId != null) {
        const parent = sorted.find((m) => m.id === prev.parentId);
        if (parent?.role === "user" && parent.parentId != null) {
          continue;
        }
        return prev.parentId;
      }
    }
    return null;
  };

  return sorted.map((message, index) => {
    if (message.role !== "assistant" || message.parentId != null) {
      return message;
    }
    const parentId = parentForOrphanAssistant(index);
    if (!parentId) {
      return message;
    }
    return { ...message, parentId };
  });
}

/** Leaf on the active branch (continued thread beats newer regen siblings). */
export function resolveHeadMessageId(
  messages: Array<
    Pick<ParentChainMessage, "id" | "parentId" | "createdAt" | "role">
  >,
): string | null {
  if (messages.length === 0) return null;
  const sorted = sortChatMessages(messages);
  const childrenOf = new Map<string | null, ParentChainMessage[]>();
  for (const message of messages) {
    const parentId = message.parentId ?? null;
    if (!childrenOf.has(parentId)) childrenOf.set(parentId, []);
    childrenOf.get(parentId)!.push(message);
  }

  const continuationUsers = sorted.filter(
    (message) => message.role === "user" && message.parentId != null,
  );
  let anchor: ParentChainMessage;
  if (continuationUsers.length > 0) {
    anchor = continuationUsers.reduce((a, b) =>
      (a.createdAt ?? 0) >= (b.createdAt ?? 0) ? a : b,
    );
  } else {
    const rootUsers = sorted.filter(
      (message) => message.role === "user" && message.parentId == null,
    );
    anchor =
      rootUsers.length > 0
        ? rootUsers[rootUsers.length - 1]
        : sorted[sorted.length - 1];
  }

  let currentId = anchor.id;
  while (true) {
    const children = childrenOf.get(currentId) ?? [];
    if (children.length === 0) return currentId;
    const next = children.reduce((a, b) =>
      (a.createdAt ?? 0) >= (b.createdAt ?? 0) ? a : b,
    );
    currentId = next.id;
  }
}

export function prepareBranchedMessagesForImport(
  messages: MessageRecord[],
): MessageRecord[] {
  const sequential = inferSequentialParentIds(messages);
  const repaired = repairAssistantParentIds(sequential);
  return orderByParentChain(repaired, { includeSiblings: true });
}
