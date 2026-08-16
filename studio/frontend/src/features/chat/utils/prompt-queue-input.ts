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
 */
export function isPromptQueueChord(event: {
  key: string;
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
}): boolean {
  if (event.key !== "Enter" || event.shiftKey) return false;
  return event.metaKey || event.ctrlKey;
}
