import type { MessageRecord } from "../types";

const ROLE_ORDER: Record<MessageRecord["role"], number> = {
  system: 0,
  user: 1,
  assistant: 2,
};

function chronologicalMessageOrder(a: MessageRecord, b: MessageRecord) {
  if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
  const roleDifference = ROLE_ORDER[a.role] - ROLE_ORDER[b.role];
  if (roleDifference !== 0) return roleDifference;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * Produces an import-safe history order for Assistant UI's message repository.
 *
 * Parented histories already carry the backend's canonical conversation order.
 * Their timestamps cannot be used as the primary ordering key: Rag Platform's
 * prologue/session timestamps are milliseconds while individual completion
 * timestamps may be seconds and user entries may omit a timestamp entirely.
 * Re-sorting that chain can put a child before its parent and make the entire
 * repository import fail. This stable topological pass preserves canonical
 * order while repairing missing/cyclic legacy parents into a linear fallback.
 */
export function prepareChatHistoryMessages(messages: MessageRecord[]): {
  hasParentIds: boolean;
  messages: MessageRecord[];
} {
  const hasParentIds = messages.some((message) => message.parentId != null);
  if (!hasParentIds) {
    return {
      hasParentIds: false,
      messages: [...messages].sort(chronologicalMessageOrder),
    };
  }

  const knownIds = new Set(messages.map((message) => message.id));
  const emittedIds = new Set<string>();
  const pending = [...messages];
  const ordered: MessageRecord[] = [];

  while (pending.length > 0) {
    const nextIndex = pending.findIndex((message) => {
      const parentId = message.parentId;
      return (
        parentId == null ||
        !knownIds.has(parentId) ||
        emittedIds.has(parentId)
      );
    });
    if (nextIndex === -1) {
      // Corrupt/cyclic legacy data must not blank the whole conversation.
      ordered.push(...pending);
      break;
    }
    const [message] = pending.splice(nextIndex, 1);
    ordered.push(message!);
    emittedIds.add(message!.id);
  }

  const normalizedIds = new Set<string>();
  let previousId: string | null = null;
  return {
    hasParentIds: true,
    messages: ordered.map((message) => {
      const parentId =
        message.parentId != null && normalizedIds.has(message.parentId)
          ? message.parentId
          : previousId;
      normalizedIds.add(message.id);
      previousId = message.id;
      return parentId === message.parentId ? message : { ...message, parentId };
    }),
  };
}
