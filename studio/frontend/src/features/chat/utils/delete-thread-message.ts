// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CompleteAttachment,
  ExportedMessageRepository,
  ThreadMessage,
} from "@assistant-ui/react";
import { db } from "@/features/chat/db";
import type { MessageRecord } from "@/features/chat/types";

function cloneContent(content: ThreadMessage["content"]): ThreadMessage["content"] {
  if (typeof content === "string") {
    return content;
  }
  return Array.isArray(content) ? JSON.parse(JSON.stringify(content)) : [];
}

function cloneAttachments(
  attachments: readonly CompleteAttachment[] | undefined,
): readonly CompleteAttachment[] {
  if (!Array.isArray(attachments)) {
    return [];
  }
  return JSON.parse(JSON.stringify(attachments));
}

function exportedItemToRecord(
  threadId: string,
  parentId: string | null,
  message: ThreadMessage,
): MessageRecord {
  const content = cloneContent(message.content);
  if (message.role === "user") {
    const attachments = cloneAttachments(message.attachments);
    const custom = message.metadata?.custom;
    return {
      id: message.id,
      threadId,
      parentId: parentId ?? null,
      role: "user",
      content: content as Extract<ThreadMessage, { role: "user" }>["content"],
      ...(attachments.length > 0 && { attachments }),
      ...(custom && Object.keys(custom).length > 0 && { metadata: custom }),
      createdAt: message.createdAt?.getTime?.() ?? Date.now(),
    };
  }
  const custom = (message.metadata?.custom ?? {}) as Record<string, unknown>;
  return {
    id: message.id,
    threadId,
    parentId: parentId ?? null,
    role: "assistant",
    content: content as Extract<ThreadMessage, { role: "assistant" }>["content"],
    ...(Object.keys(custom).length > 0 && { metadata: custom }),
    createdAt: message.createdAt?.getTime?.() ?? Date.now(),
  };
}

/**
 * Persist the exact message list represented by `exp` for this thread, removing
 * Dexie rows that are no longer present (e.g. after a delete).
 */
async function syncExportedRepositoryToDexie(
  remoteId: string,
  exp: ExportedMessageRepository,
): Promise<void> {
  await db.transaction("rw", db.messages, async () => {
    const keepIds = new Set(exp.messages.map((x) => x.message.id));
    const existing = await db.messages.where("threadId").equals(remoteId).toArray();
    const idsToDelete = existing
      .filter((m) => !keepIds.has(m.id))
      .map((m) => m.id);
    if (idsToDelete.length > 0) {
      await db.messages.bulkDelete(idsToDelete);
    }
    await db.messages.bulkPut(
      exp.messages.map(({ message, parentId }) =>
        exportedItemToRecord(remoteId, parentId, message),
      ),
    );
  });
}

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

type ExportedMessageItem = ExportedMessageRepository["messages"][number];

function collectSubtreeIds(
  childrenByParentId: Map<string | null, string[]>,
  rootId: string,
): Set<string> {
  const ids = new Set<string>();
  const stack = [rootId];

  while (stack.length > 0) {
    const currentId = stack.pop();
    if (!currentId || ids.has(currentId)) {
      continue;
    }
    ids.add(currentId);
    const children = childrenByParentId.get(currentId) ?? [];
    for (const childId of children) {
      stack.push(childId);
    }
  }

  return ids;
}

function getSurvivingChildren(
  childrenByParentId: Map<string | null, string[]>,
  deletedIds: Set<string>,
  parentId: string | null,
): string[] {
  return (childrenByParentId.get(parentId) ?? []).filter((id) => !deletedIds.has(id));
}

function findPreferredLeaf(
  childrenByParentId: Map<string | null, string[]>,
  deletedIds: Set<string>,
  rootId: string,
): string {
  let currentId = rootId;

  while (true) {
    const children = getSurvivingChildren(childrenByParentId, deletedIds, currentId);
    if (children.length === 0) {
      return currentId;
    }
    currentId = children[children.length - 1]!;
  }
}

function findReplacementHeadId(
  childrenByParentId: Map<string | null, string[]>,
  deletedIds: Set<string>,
  startParentId: string | null,
): string | null {
  if (startParentId === null) {
    const roots = getSurvivingChildren(childrenByParentId, deletedIds, null);
    return roots.length > 0
      ? findPreferredLeaf(childrenByParentId, deletedIds, roots[roots.length - 1]!)
      : null;
  }

  const siblings = getSurvivingChildren(childrenByParentId, deletedIds, startParentId);
  if (siblings.length > 0) {
    return findPreferredLeaf(
      childrenByParentId,
      deletedIds,
      siblings[siblings.length - 1]!,
    );
  }

  return startParentId;
}

/**
 * Delete a message from exported history without leaving malformed user->user /
 * assistant->assistant adjacency behind.
 *
 * The assistant delete button is only shown on assistant messages, so deleting
 * an assistant removes that assistant branch and its descendants while keeping
 * sibling assistant branches under the same user prompt intact.
 *
 * This avoids assistant-ui's default reparenting behavior, which can turn
 * `user -> assistant -> user -> assistant` into `user -> user -> assistant`
 * after deleting an assistant message.
 */
export function deleteMessageFromExportedRepository(
  exported: ExportedMessageRepository,
  messageId: string,
): ExportedMessageRepository {
  const messageById = new Map<string, ExportedMessageItem>();
  const childrenByParentId = new Map<string | null, string[]>();

  for (const item of exported.messages) {
    messageById.set(item.message.id, item);
    const siblings = childrenByParentId.get(item.parentId) ?? [];
    siblings.push(item.message.id);
    childrenByParentId.set(item.parentId, siblings);
  }

  const target = messageById.get(messageId);
  if (!target) {
    throw new Error(`Message ${messageId} not found in exported thread history.`);
  }

  const idsToDelete = collectSubtreeIds(childrenByParentId, target.message.id);
  const nextHeadId =
    exported.headId && !idsToDelete.has(exported.headId)
      ? exported.headId
      : findReplacementHeadId(childrenByParentId, idsToDelete, target.parentId);

  return {
    headId: nextHeadId,
    messages: exported.messages.filter((item) => !idsToDelete.has(item.message.id)),
  };
}

/**
 * Remove a message from the thread and mirror the result to IndexedDB.
 */
export async function deleteThreadMessage(args: {
  thread: ThreadImportExport;
  messageId: string;
  remoteId: string | undefined;
}): Promise<void> {
  const { thread, messageId, remoteId } = args;
  const exported = thread.export();
  const next = deleteMessageFromExportedRepository(exported, messageId);
  if (remoteId) {
    await syncExportedRepositoryToDexie(remoteId, next);
  }
  thread.import(next);
}
