// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ParentLinkedMessage = {
  id: string;
  parentId?: string | null;
  createdAt?: number;
};

// follow the newest parent chain because response slots can predate the next user message.
export function orderByParentChain<T extends ParentLinkedMessage>(
  messages: T[],
  options: { includeSiblings?: boolean } = {},
): T[] {
  const { includeSiblings = true } = options;
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

  if (includeSiblings) {
    for (const message of byId.values()) {
      result.push(message);
    }
  }
  return result;
}
