// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  Attachment,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";
import { cachedTextAttachment } from "../text-attachment-accept.ts";
import { attachmentContentText, isPastedTextFile } from "./pasted-text.ts";

import { annotationsContentText, annotationsOfFile } from "./document-annotations.ts";

export type QueuedPrompt = {
  prompt: string;
  attachments?: CompleteAttachment[];
};

/** Shared by normal sends and the queue so filenames, encoding and paste markers agree. */
export function completeTextAttachment(
  attachment: PendingAttachment,
  text: string,
): CompleteAttachment {
  const annotations = annotationsOfFile(attachment.file);
  return {
    id: attachment.id,
    type: "document",
    name: attachment.name,
    contentType: attachment.contentType,
    content: [
      {
        type: "text",
        text: annotations ? annotationsContentText(annotations) : attachmentContentText(
          attachment.name,
          text,
          isPastedTextFile(attachment.file),
          attachment.file.size,
        ),
      },
    ],
    status: { type: "complete" },
  };
}

/** Require both successful adapter validation and decoded text. Extension alone cannot
 * authorize a pending, failed, binary, or differently handled document. */
export function canQueueTextAttachment(attachment: Attachment): boolean {
  return (
    attachment.type === "document" &&
    attachment.status.type === "requires-action" &&
    attachment.status.reason === "composer-send" &&
    attachment.file !== undefined &&
    cachedTextAttachment(attachment.file) !== undefined
  );
}

/** Snapshot prepared contents without retaining File objects or awaiting a read that
 * could let a later submit overtake this one. Reject mixed unsupported attachments. */
export function snapshotQueuedTextAttachments(
  attachments: readonly Attachment[],
): CompleteAttachment[] | null {
  if (!attachments.length || !attachments.every(canQueueTextAttachment))
    return null;
  return attachments.map((attachment) =>
    completeTextAttachment(
      attachment as PendingAttachment,
      cachedTextAttachment(attachment.file!)!,
    ),
  );
}

export function normalizeQueuedPrompt(
  item: string | QueuedPrompt,
): QueuedPrompt {
  return typeof item === "string"
    ? { prompt: item.trim() }
    : { ...item, prompt: item.prompt.trim() };
}

export function queuedPromptHasContent(item: QueuedPrompt): boolean {
  return item.prompt.trim().length > 0 || Boolean(item.attachments?.length);
}

export function queuedPromptMessage(item: QueuedPrompt) {
  return {
    role: "user" as const,
    content: item.prompt ? [{ type: "text" as const, text: item.prompt }] : [],
    attachments: item.attachments ?? [],
    createdAt: new Date(),
  };
}
