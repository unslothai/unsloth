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
  attachmentFiles?: File[];
};

export async function prepareQueuedPromptFiles(
  item: QueuedPrompt,
  prepare: (
    file: File,
    attachment: CompleteAttachment,
  ) => Promise<CompleteAttachment>,
): Promise<QueuedPrompt> {
  const { attachments, attachmentFiles } = item;
  if (attachmentFiles === undefined || attachmentFiles.length === 0) {
    return item;
  }
  if (!attachments || attachments.length !== attachmentFiles.length) {
    throw new Error("Queued attachment snapshot is inconsistent");
  }
  return {
    ...item,
    attachments: await Promise.all(
      attachments.map((attachment, index) => {
        const file = attachmentFiles[index];
        if (!file) {
          throw new Error("Queued attachment snapshot is inconsistent");
        }
        return prepare(file, attachment);
      }),
    ),
    attachmentFiles: undefined,
  };
}

/** keeps normal sends and queues aligned on filenames, encoding, and paste markers. */
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

/** requires a validated decode because extensions can also match pending, failed, or binary files. */
export function canQueueTextAttachment(attachment: Attachment): boolean {
  return (
    attachment.type === "document" &&
    attachment.status.type === "requires-action" &&
    attachment.status.reason === "composer-send" &&
    attachment.file !== undefined &&
    cachedTextAttachment(attachment.file) !== undefined
  );
}

/** snapshots decoded content without File objects or reads that later submits could overtake. */
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

export function snapshotQueuedTextPrompt(
  prompt: string,
  attachments: readonly Attachment[],
): QueuedPrompt | null {
  const completed = snapshotQueuedTextAttachments(attachments);
  if (!completed) {
    return null;
  }
  return {
    prompt,
    attachments: completed,
    attachmentFiles: attachments.map((attachment) => {
      if (!attachment.file) {
        throw new Error("Queued attachment snapshot is inconsistent");
      }
      return attachment.file;
    }),
  };
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
