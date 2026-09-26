// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CompleteAttachment, PendingAttachment } from "@assistant-ui/react";
import { uploadChatAttachmentOriginal } from "./api/chat-api";

/** A sent document's original file, kept by the server (core/chat_originals.py). */
export interface ChatAttachmentOriginal {
  sha256: string;
  sizeBytes: number;
}

// As the server keeps: the documents a viewer shows as pages, a grid or slides.
const ORIGINAL_EXTENSIONS = /\.(pdf|docx|xlsx|xlsm|pptx)$/i;
// A document picked by type alone still keeps its original, named for the server's extension check.
const ORIGINAL_TYPES: Record<string, string> = {
  "application/pdf": "pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
  "application/vnd.ms-excel.sheet.macroEnabled.12": "xlsm",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
};

function originalUpload(file: File): File | null {
  if (ORIGINAL_EXTENSIONS.test(file.name)) return file;
  const type = file.type.split(";", 1)[0]!.trim().toLowerCase();
  const extension = Object.hasOwn(ORIGINAL_TYPES, type) ? ORIGINAL_TYPES[type] : undefined;
  return extension ? new File([file], `${file.name || "document"}.${extension}`, { type: file.type }) : null;
}

export function attachmentOriginal(attachment: unknown): ChatAttachmentOriginal | null {
  const original = (attachment as { original?: unknown } | null)?.original;
  if (!original || typeof original !== "object") return null;
  const { sha256, sizeBytes } = original as Partial<ChatAttachmentOriginal>;
  return typeof sha256 === "string" && typeof sizeBytes === "number" ? { sha256, sizeBytes } : null;
}

/** Adds the kept original to a sent document. In temporary chats the file stays in memory instead; `temporary` and `epoch`
 *  (the auth session) are as they were when the send began. Upload errors are ignored. */
export async function withAttachmentOriginal(
  pending: PendingAttachment,
  complete: CompleteAttachment,
  temporary: boolean,
  epoch: number,
): Promise<CompleteAttachment> {
  const upload = complete.type === "document" ? originalUpload(pending.file) : null;
  if (!upload) return complete;
  // Kept in memory only, so it still opens as a document; uploaded if the chat is saved.
  if (temporary) return { ...complete, file: pending.file } as CompleteAttachment;
  try {
    const original: ChatAttachmentOriginal = await uploadChatAttachmentOriginal(upload, epoch);
    return { ...complete, original } as CompleteAttachment;
  } catch {
    return complete;
  }
}

/** For a temporary chat being saved: uploads each document's in-memory file as its original, and
 *  drops the file, which does not serialize. A failed upload only affects viewing. */
export async function persistAttachmentOriginals(
  attachments: readonly CompleteAttachment[] | undefined,
  epoch: number,
): Promise<CompleteAttachment[]> {
  return Promise.all(
    (attachments ?? []).map(async (attachment) => {
      const { file, ...rest } = attachment as CompleteAttachment & { file?: unknown };
      if (file === undefined) return attachment;
      const upload = file instanceof File && !attachmentOriginal(rest) ? originalUpload(file) : null;
      if (!upload) return rest as CompleteAttachment;
      try {
        const original: ChatAttachmentOriginal = await uploadChatAttachmentOriginal(upload, epoch);
        return { ...rest, original } as CompleteAttachment;
      } catch {
        return rest as CompleteAttachment;
      }
    }),
  );
}
