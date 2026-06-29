// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * assistant-ui exposes no public `deleteMessage` in our version, but
 * `MessageRepository` already does branch-safe deletion. We import it from
 * `@assistant-ui/core/internal` (the exported internal surface); avoid the
 * deeper `runtime/utils/message-repository` path since newer releases no
 * longer export arbitrary deep paths.
 *
 * Keep this file the only importer of `MessageRepository`. When bumping
 * `@assistant-ui/react` / `core`, re-run chat delete + reload smoke tests;
 * the path or API may change without a semver signal.
 */
import { MessageRepository } from "@assistant-ui/core/internal";
import type {
  CompleteAttachment,
  ExportedMessageRepository,
  ThreadMessage,
} from "@assistant-ui/react";
import type { MessageRecord } from "../types";
import {
  ensureStoredChatThread,
  syncStoredChatMessages,
} from "./chat-history-storage";

function cloneContent(
  content: ThreadMessage["content"],
): ThreadMessage["content"] {
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

export function exportedItemToRecord(
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
    content: content as Extract<
      ThreadMessage,
      { role: "assistant" }
    >["content"],
    ...(Object.keys(custom).length > 0 && { metadata: custom }),
    createdAt: message.createdAt?.getTime?.() ?? Date.now(),
  };
}

/**
 * Persist exported messages, pruning only for explicit delete flows.
 */
export async function syncExportedRepositoryToBackend(
  remoteId: string,
  exp: ExportedMessageRepository,
  options: { pruneMissing?: boolean } = {},
): Promise<void> {
  await ensureStoredChatThread(remoteId);
  await syncStoredChatMessages(
    remoteId,
    exp.messages.map(({ message, parentId }) =>
      exportedItemToRecord(remoteId, parentId, message),
    ),
    { pruneMissing: options.pruneMissing },
  );
}

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

/**
 * Remove a message from the thread and mirror the result to backend storage.
 */
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
    });
  }
  thread.import(next);
}
