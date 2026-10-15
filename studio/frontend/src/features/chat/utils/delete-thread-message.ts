// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Internal assistant-ui import; keep this the only importer and re-test delete on upgrades. */
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

// Epoch millis pass through; `getTime?.()` alone re-dated them to now.
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
  await ensureStoredChatThread(remoteId);
  // Use the backend copy and surface read failures; an unreconciled payload is rejected wholesale.
  const stored = await listChatMessages(remoteId).catch((error: unknown) => {
    throw new Error(
      `Could not read the stored research messages for thread ${remoteId} before syncing`,
      { cause: error },
    );
  });
  return reconcileServerManagedMessages(records, stored);
}

export async function syncExportedRepositoryToBackend(
  remoteId: string,
  exp: ExportedMessageRepository,
  options: { pruneMissing?: boolean; deletedMessageIds?: string[] } = {},
): Promise<void> {
  // syncStoredChatMessages ensures the row itself.
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
