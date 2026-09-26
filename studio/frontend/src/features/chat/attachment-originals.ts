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

/** Adds the kept original to a sent document. Skipped in temporary chats (`temporary`, as it was
 *  when the send began); upload errors are ignored. */
export async function withAttachmentOriginal(
  pending: PendingAttachment,
  complete: CompleteAttachment,
  temporary: boolean,
): Promise<CompleteAttachment> {
  const upload = complete.type === "document" && !temporary ? originalUpload(pending.file) : null;
  if (!upload) return complete;
  try {
    const original: ChatAttachmentOriginal = await uploadChatAttachmentOriginal(upload);
    return { ...complete, original } as CompleteAttachment;
  } catch {
    return complete;
  }
}
