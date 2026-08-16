// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** How keyboard and drag input map to prompt-queue actions. */

/**
 * Private drag type for a queue row, so a reorder is distinguishable from a
 * file or a text selection dragged in from outside.
 */
export const PROMPT_QUEUE_DRAG_TYPE = "application/x-unsloth-prompt-queue-item";

/**
 * Whether a drag carries a queue row. Anything else must pass through
 * untouched: the page dropzone skips events the row already prevented, so
 * claiming a file drag here would silently drop the file.
 */
export function isPromptQueueDragTypes(
  types: ArrayLike<string> | undefined | null,
): boolean {
  if (!types) return false;
  return Array.from(types).includes(PROMPT_QUEUE_DRAG_TYPE);
}

/**
 * Cmd/Ctrl+Enter, the queue chord. Shift+Enter is a newline whatever else is
 * held, so Shift disqualifies it.
 *
 * Alt disqualifies it too, because Windows reports AltGr as Ctrl+Alt on any
 * layout the browser does not recognise as having a real AltGraph. Layouts that
 * need AltGr for everyday characters would otherwise queue on a keypress the
 * user meant as something else, and there is no Ctrl+Alt+Enter binding here
 * worth keeping. `key` rather than `code`, so the numeric keypad's Enter is the
 * same chord.
 */
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

/**
 * Whether a queue start for this thread is already registered but has not yet
 * reached the queue store. Starting one awaits settings hydration, and during
 * that gap nothing else marks the thread as queueing, so a plain Enter would
 * take the send path and the pending queue would then dispatch its own copy of
 * the same prompt. Treat the thread as queueing until the start resolves.
 */
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

/**
 * Identity of a pasted-text queue start, held for the length of the file read
 * that precedes it. A submit during the read is routed to the queue branch, so
 * without this it would start a second read of the same attachment and queue a
 * duplicate once the first read's reservation had been released.
 *
 * The wait mode is deliberately not part of it. It is recomputed per submit and
 * can flip if a run starts or ends mid-read, which would split one prompt
 * across two keys and reintroduce the duplicate. What identifies the prompt is
 * the thread, the text and the attachments it is assembled from.
 */
export function pastedTextQueueKey(
  threadId: string | null,
  text: string,
  attachmentIds: readonly string[],
): string {
  return JSON.stringify([threadId, text, attachmentIds]);
}
