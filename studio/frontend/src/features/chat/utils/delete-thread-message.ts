// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** assistant-ui exposes no public `deleteMessage` in our version, but `MessageRepository` already
 *  does branch-safe deletion. Imported from `@assistant-ui/core/internal`, the exported internal
 *  surface; avoid the deeper `runtime/utils/message-repository` path, which newer releases no
 *  longer export. Keep this file the only importer, and re-run chat delete plus reload smoke
 *  tests when bumping `@assistant-ui/react`: the API may change without a semver signal. */
import { MessageRepository } from "@assistant-ui/core/internal";
import type {
  CompleteAttachment,
  ExportedMessageRepository,
  ThreadMessage,
} from "@assistant-ui/react";
import { listChatMessages } from "../api/chat-api";
import type { MessageRecord } from "../types";
import {
  ensureStoredChatThread,
  syncStoredChatMessages,
} from "./chat-history-storage";
import {
  hasResearchMetadata,
  reconcileServerManagedMessages,
} from "./research-message-sync";

// A copy of the list, not of what is in it. assistant-ui replaces parts and attachments rather
// than mutating them, and the records built from these are serialized straight into the PUT
// body, so a deep clone of the whole thread bought nothing.
function snapshotContent(
  content: ThreadMessage["content"],
): ThreadMessage["content"] {
  if (typeof content === "string") {
    return content;
  }
  return Array.isArray(content)
    ? ([...content] as ThreadMessage["content"])
    : [];
}

// Epoch millis pass through; `getTime?.()` alone re-dated them to now and reordered the thread.
function toEpochMillis(value: unknown): number {
  if (value instanceof Date) return value.getTime();
  return typeof value === "number" && Number.isFinite(value) ? value : Date.now();
}

function snapshotAttachments(
  attachments: readonly CompleteAttachment[] | undefined,
): readonly CompleteAttachment[] {
  return Array.isArray(attachments) ? [...attachments] : [];
}

export function exportedItemToRecord(
  threadId: string,
  parentId: string | null,
  message: ThreadMessage,
): MessageRecord {
  const content = snapshotContent(message.content);
  if (message.role === "user") {
    const attachments = snapshotAttachments(message.attachments);
    const custom = message.metadata?.custom;
    return {
      id: message.id,
      threadId,
      parentId: parentId ?? null,
      role: "user",
      content: content as Extract<ThreadMessage, { role: "user" }>["content"],
      ...(attachments.length > 0 && { attachments }),
      ...(custom && Object.keys(custom).length > 0 && { metadata: custom }),
      createdAt: toEpochMillis(message.createdAt),
    };
  }
  const custom = (message.metadata?.custom ?? {}) as Record<string, unknown>;
  return {
    id: message.id,
    threadId,
    parentId: parentId ?? null,
    role: "assistant",
    content: content as Extract<
      ThreadMessage,
      { role: "assistant" }
    >["content"],
    ...(Object.keys(custom).length > 0 && { metadata: custom }),
    createdAt: toEpochMillis(message.createdAt),
  };
}

async function withStoredResearchMessages(
  remoteId: string,
  records: MessageRecord[],
): Promise<MessageRecord[]> {
  if (!records.some((record) => hasResearchMetadata(record.metadata))) {
    return records;
  }
  // The read below wants the row in place, which is the only reason this path ensures it.
  await ensureStoredChatThread(remoteId);
  // The backend copy, not the legacy-merged one: only what it stored can be echoed back to it.
  // Swallowing a failure here would send the unreconciled payload, which the server rejects
  // wholesale, so the read failure has to surface as itself rather than as a later 409.
  const stored = await listChatMessages(remoteId).catch((error: unknown) => {
    throw new Error(
      `Could not read the stored research messages for thread ${remoteId} before syncing`,
      { cause: error },
    );
  });
  return reconcileServerManagedMessages(records, stored);
}

/** Persist exported messages, pruning only for explicit delete flows. */
export async function syncExportedRepositoryToBackend(
  remoteId: string,
  exp: ExportedMessageRepository,
  options: { pruneMissing?: boolean; deletedMessageIds?: string[] } = {},
): Promise<void> {
  // No ensureStoredChatThread here: syncStoredChatMessages ensures the row itself, and this used
  // to make every save pay for the same GET /threads/{id} twice.
  const records = exp.messages.map(({ message, parentId }) =>
    exportedItemToRecord(remoteId, parentId, message),
  );
  await syncStoredChatMessages(
    remoteId,
    await withStoredResearchMessages(remoteId, records),
    {
      pruneMissing: options.pruneMissing,
      deletedMessageIds: options.deletedMessageIds,
    },
  );
}

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

/** Remove a message from the thread and mirror the result to backend storage. */
export async function deleteThreadMessage(args: {
  thread: ThreadImportExport;
  messageId: string;
  remoteId: string | undefined;
}): Promise<void> {
  const { thread, messageId, remoteId } = args;
  const exported = thread.export();
  const repo = new MessageRepository();
  repo.import(exported);

  const target = exported.messages.find(
    ({ message }) => message.id === messageId,
  );
  const assistantReplyIds =
    target?.message.role === "user"
      ? exported.messages
          .filter(
            ({ parentId, message }) =>
              parentId === messageId && message.role === "assistant",
          )
          .map(({ message }) => message.id)
      : [];

  // Delete the prompt first; that relinks its replies up to the prompt's parent
  repo.deleteMessage(messageId);
  for (const replyId of assistantReplyIds) {
    repo.deleteMessage(replyId);
  }

  const next = repo.export();
  if (remoteId) {
    await syncExportedRepositoryToBackend(remoteId, next, {
      pruneMissing: true,
      deletedMessageIds: [messageId, ...assistantReplyIds],
    });
  }
  thread.import(next);
}
