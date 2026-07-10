// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Notifies loaded chat runtimes when the Data tab deletes a stored attachment.
 * Without this, the active thread's in-memory repository still holds the
 * attachment, and any later repo-to-storage sync (e.g. deleting a message in
 * that thread) writes it back, undoing the deletion.
 */

export type ChatAttachmentDeletedEvent = {
  messageId: string;
  attachmentId: string;
};

type Listener = (event: ChatAttachmentDeletedEvent) => void;

const listeners = new Set<Listener>();

export function onChatAttachmentDeleted(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function emitChatAttachmentDeleted(
  event: ChatAttachmentDeletedEvent,
): void {
  for (const listener of [...listeners]) {
    listener(event);
  }
}
