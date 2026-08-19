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
 * Cmd/Ctrl+Enter, the queue chord. Shift+Enter stays a newline whatever else is
 * held, and Alt disqualifies it too: Windows reports AltGr as Ctrl+Alt on any
 * layout the browser does not read as having a real AltGraph, and those layouts
 * hold AltGr for everyday characters. `key`, not `code`, so the keypad's Enter
 * is the same chord.
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
 * Whether a queue start for this thread is registered but has not reached the
 * queue store yet. Starting one awaits settings hydration, and nothing else
 * marks the thread as queueing during that gap, so a plain Enter would send the
 * same text the pending queue is about to dispatch.
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
 * Whether a normal attachment can use the queue button while a response is
 * running. Pasted-text attachments have their own queue path that turns them
 * back into prompt text; every other attachment must stay in the composer and
 * be sent through the attachment adapter once the current run is idle.
 */
export function isAttachmentQueueable(state: {
  hasAttachments: boolean;
  attachmentsAreAllPastedText: boolean;
  hasPendingAudio: boolean;
  isComposing: boolean;
  hasPendingAttachments: boolean;
  hasMaterializingImageAttachments: boolean;
  hasMaterializingAudioAttachments: boolean;
  disabled: boolean;
  overlay: boolean;
}): boolean {
  return (
    state.hasAttachments &&
    !state.attachmentsAreAllPastedText &&
    !state.hasPendingAudio &&
    !state.isComposing &&
    !state.hasPendingAttachments &&
    !state.hasMaterializingImageAttachments &&
    !state.hasMaterializingAudioAttachments &&
    !state.disabled &&
    !state.overlay
  );
}

/**
 * Identity of a pasted-text queue start, held for the length of the file read
 * before it, so a submit during that read joins it instead of starting a second
 * read of the same attachment.
 *
 * The wait mode is deliberately out: it is recomputed per submit and flips if a
 * run starts mid-read, which would split one prompt across two keys.
 */
export function pastedTextQueueKey(
  threadId: string | null,
  text: string,
  attachmentIds: readonly string[],
): string {
  return JSON.stringify([threadId, text, attachmentIds]);
}
