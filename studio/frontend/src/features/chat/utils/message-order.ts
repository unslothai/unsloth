// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ParentLinkedMessage = {
  id: string;
  parentId?: string | null;
  createdAt?: number;
  role?: string;
};

const ROLE_ORDER: Record<string, number> = { system: 0, user: 1, assistant: 2 };

// Rows in storage order. A null is a root only after a recorded parent; earlier it chains.
export function createParentResolver(): (
  message: ParentLinkedMessage,
) => string | null {
  let previousId: string | null = null;
  let sawRecordedParent = false;
  return (message) => {
    const inferred =
      message.parentId === undefined ||
      (message.parentId === null && !sawRecordedParent);
    const parentId = inferred ? previousId : message.parentId;
    if (message.parentId != null) {
      sawRecordedParent = true;
    }
    previousId = message.id;
    return parentId ?? null;
  };
}

export function compareStoredMessages(
  a: ParentLinkedMessage,
  b: ParentLinkedMessage,
): number {
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
}

export function orderBySelectedBranch<T extends ParentLinkedMessage>(
  messages: T[],
  headId?: string | null,
): T[] {
  const sorted = messages.slice().sort(compareStoredMessages);

  const byId = new Map<string, T>();
  const parentOf = new Map<string, string | null>();
  const resolveParent = createParentResolver();
  for (const message of sorted) {
    byId.set(message.id, message);
    parentOf.set(message.id, resolveParent(message));
  }

  const chain: T[] = [];
  const seen = new Set<string>();
  // undefined follows the newest message; null is a live branch with nothing saved yet.
  let currentId: string | null =
    headId === undefined ? (sorted.at(-1)?.id ?? null) : headId;
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

// remote turns can extend a saved head; use the newest leaf because assistant-ui drops descendants.
export function resolveSavedBranchHead<T extends ParentLinkedMessage>(
  messages: T[],
  savedHeadId: string | null | undefined,
): string | undefined {
  if (!savedHeadId) return undefined;
  const sorted = messages.slice().sort(compareStoredMessages);
  const resolveParent = createParentResolver();
  const children = new Map<string, string[]>();
  const order = new Map<string, number>();
  sorted.forEach((message, index) => {
    order.set(message.id, index);
    const parentId = resolveParent(message);
    if (parentId == null) return;
    const siblings = children.get(parentId) ?? [];
    siblings.push(message.id);
    children.set(parentId, siblings);
  });
  if (!order.has(savedHeadId)) return undefined;
  let headId = savedHeadId;
  let headOrder = -1;
  const seen = new Set<string>();
  const pending = [savedHeadId];
  while (pending.length > 0) {
    const id = pending.pop()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const below = children.get(id);
    if (below?.length) {
      pending.push(...below);
    } else if ((order.get(id) ?? -1) > headOrder) {
      headId = id;
      headOrder = order.get(id) ?? -1;
    }
  }
  return headId;
}

// follow the newest parent chain because response slots can predate the next user message.
export function orderByParentChain<T extends ParentLinkedMessage>(
  messages: T[],
  options: { includeSiblings?: boolean; headId?: string | null } = {},
): T[] {
  const { includeSiblings = true, headId } = options;
  if (!includeSiblings) {
    return orderBySelectedBranch(messages, headId);
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
