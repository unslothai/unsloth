// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ParentLinkedMessage = {
  id: string;
  parentId?: string | null;
  createdAt?: number;
  role?: string;
};

const ROLE_ORDER: Record<string, number> = { system: 0, user: 1, assistant: 2 };

export function orderBySelectedBranch<T extends ParentLinkedMessage>(
  messages: T[],
): T[] {
  const sorted = messages.slice().sort((a, b) => {
    const createdAtDelta = (a.createdAt ?? 0) - (b.createdAt ?? 0);
    if (createdAtDelta !== 0) {
      return createdAtDelta;
    }
    const roleDelta =
      (ROLE_ORDER[a.role ?? ""] ?? 99) - (ROLE_ORDER[b.role ?? ""] ?? 99);
    if (roleDelta !== 0) {
      return roleDelta;
    }
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });

  const byId = new Map<string, T>();
  const parentOf = new Map<string, string | null>();
  let previousId: string | null = null;
  for (const message of sorted) {
    byId.set(message.id, message);
    parentOf.set(message.id, message.parentId ?? previousId);
    previousId = message.id;
  }

  const chain: T[] = [];
  const seen = new Set<string>();
  let currentId: string | null = sorted.at(-1)?.id ?? null;
  while (currentId != null && !seen.has(currentId)) {
    seen.add(currentId);
    const message = byId.get(currentId);
    if (!message) {
      break;
    }
    chain.push(message);
    currentId = parentOf.get(currentId) ?? null;
  }
  return chain.reverse();
}

// follow the newest parent chain because response slots can predate the next user message.
export function orderByParentChain<T extends ParentLinkedMessage>(
  messages: T[],
  options: { includeSiblings?: boolean } = {},
): T[] {
  const { includeSiblings = true } = options;
  if (!includeSiblings) {
    return orderBySelectedBranch(messages);
  }
  const byId = new Map<string, T>(
    messages.map((message) => [message.id, message]),
  );
  const childrenOf = new Map<string | null, T[]>();
  for (const message of messages) {
    const parentId = message.parentId ?? null;
    const children = childrenOf.get(parentId) ?? [];
    children.push(message);
    childrenOf.set(parentId, children);
  }

  const result: T[] = [];
  let currentId: string | null = null;
  while (childrenOf.has(currentId)) {
    const children: T[] = childrenOf.get(currentId) ?? [];
    const next: T = children.reduce((latest: T, candidate: T) =>
      (latest.createdAt ?? 0) >= (candidate.createdAt ?? 0)
        ? latest
        : candidate,
    );
    result.push(next);
    currentId = next.id;
    byId.delete(next.id);
  }

  for (const message of byId.values()) {
    result.push(message);
  }
  return result;
}
