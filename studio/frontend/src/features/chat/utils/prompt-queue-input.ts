// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const PROMPT_QUEUE_DRAG_TYPE = "application/x-unsloth-prompt-queue-item";

/** Never claim other drags: the page dropzone skips prevented events, dropping the file. */
export function isPromptQueueDragTypes(
  types: ArrayLike<string> | undefined | null,
): boolean {
  if (!types) return false;
  return Array.from(types).includes(PROMPT_QUEUE_DRAG_TYPE);
}

/** Alt disqualifies it: Windows reports AltGr as Ctrl+Alt. */
export function isPromptQueueChord(event: {
  key: string;
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey?: boolean;
}): boolean {
  if (event.key !== "Enter" || event.shiftKey || event.altKey) return false;
  return event.metaKey || event.ctrlKey;
}

/** Starting a queue awaits hydration; plain Enter must not send the text in that gap. */
export function hasPendingPromptQueueStart(
  reservations: Iterable<{ cancelled: boolean; threadId: string | null }>,
  threadId: string | null,
): boolean {
  for (const reservation of reservations) {
    if (reservation.cancelled) continue;
    if (reservation.threadId === threadId) return true;
  }
  return false;
}

/** Excludes the wait mode, which can flip mid-read and split one prompt across two keys. */
export function pastedTextQueueKey(
  threadId: string | null,
  text: string,
  attachmentIds: readonly string[],
): string {
  return JSON.stringify([threadId, text, attachmentIds]);
}
