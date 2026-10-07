// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CompleteAttachment, PendingAttachment } from "@assistant-ui/react";
import { uploadChatAttachmentOriginal } from "./api/chat-api";

export interface ChatAttachmentOriginal {
  sha256: string;
  sizeBytes: number;
}

const ORIGINAL_EXTENSIONS = /\.(pdf|docx|xlsx|xlsm|pptx)$/i;
const ORIGINAL_TYPES: Record<string, string> = {
  "application/pdf": "pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
  "application/vnd.ms-excel.sheet.macroenabled.12": "xlsm",
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

export async function withAttachmentOriginal(
  pending: PendingAttachment,
  complete: CompleteAttachment,
  temporary: boolean,
  epoch: number,
  forPythonTool: boolean,
): Promise<CompleteAttachment> {
  // Other documents are uploaded only for the python tool's sandbox.
  const upload =
    complete.type === "document" && !attachmentOriginal(complete)
      ? (originalUpload(pending.file) ?? (forPythonTool ? pending.file : null))
      : null;
  if (!upload) return complete;
  // A temporary chat keeps its files in memory, python tool or not: it promises nothing is saved.
  if (temporary) return { ...complete, file: pending.file } as CompleteAttachment;
  try {
    const original: ChatAttachmentOriginal = await uploadChatAttachmentOriginal(upload, epoch);
    return { ...complete, original } as CompleteAttachment;
  } catch {
    return complete;
  }
}

export async function persistAttachmentOriginals(
  attachments: readonly CompleteAttachment[] | undefined,
  epoch: number,
): Promise<CompleteAttachment[]> {
  return Promise.all(
    (attachments ?? []).map(async (attachment) => {
      const { file, ...rest } = attachment as CompleteAttachment & { file?: unknown };
      if (file === undefined) return attachment;
      // A file was kept only because a send wanted it stored (a python turn keeps any document).
      const upload =
        file instanceof File && !attachmentOriginal(rest) ? (originalUpload(file) ?? file) : null;
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

// Unreferenced originals are swept after an hour; re-upload at half that to restart the clock.
const STAGED_UPLOAD_MAX_AGE_MS = 30 * 60 * 1000;

export function reuseStagedUpload(stagedAt: number | undefined, now = Date.now()): boolean {
  return stagedAt !== undefined && now - stagedAt < STAGED_UPLOAD_MAX_AGE_MS;
}
