// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CompleteAttachment, PendingAttachment } from "@assistant-ui/react";
import { uploadChatAttachmentOriginal } from "./api/chat-api";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

/** A sent document's original file, kept by the server (core/chat_originals.py). */
export interface ChatAttachmentOriginal {
  sha256: string;
  sizeBytes: number;
}

// As the server keeps: the documents a viewer shows as pages, a grid or slides.
const ORIGINAL_EXTENSIONS = /\.(pdf|docx|xlsx|xlsm|pptx)$/i;

export function attachmentOriginal(attachment: unknown): ChatAttachmentOriginal | null {
  const original = (attachment as { original?: unknown } | null)?.original;
  if (!original || typeof original !== "object") return null;
  const { sha256, sizeBytes } = original as Partial<ChatAttachmentOriginal>;
  return typeof sha256 === "string" && typeof sizeBytes === "number" ? { sha256, sizeBytes } : null;
}

/** Adds the kept original to a sent document. Skipped in temporary chats; upload errors are ignored. */
export async function withAttachmentOriginal(
  pending: PendingAttachment,
  complete: CompleteAttachment,
): Promise<CompleteAttachment> {
  const file = pending.file;
  if (complete.type !== "document" || !ORIGINAL_EXTENSIONS.test(file.name)) return complete;
  if (useChatRuntimeStore.getState().incognito) return complete;
  try {
    const original: ChatAttachmentOriginal = await uploadChatAttachmentOriginal(file);
    return { ...complete, original } as CompleteAttachment;
  } catch {
    return complete;
  }
}
