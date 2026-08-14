// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a write into the composer is the sent message coming back.
 *
 * Sending clears the composer, but a write carrying the pre-send text can still
 * land after it: an input or compositionend event the engine queued against the
 * old value (autocorrect commit, IME finalise, undo), or a draft autosave that
 * raced the send. Applying it refills the composer and the next autosave makes
 * that stick, so the prompt sends and the text stays in the box.
 *
 * A refusal keeps the sent text, since an engine can queue several such events.
 * Any write that differs drops it, so typing is never held up.
 */
export function applySentTextGuard(
  /** Text the last send took, or null when no send is being guarded. */
  sentText: string | null,
  /** Value the write wants to put into the composer. */
  value: string,
): { accept: boolean; sentText: string | null } {
  if (sentText === null) return { accept: true, sentText: null };
  if (value === sentText) return { accept: false, sentText };
  return { accept: true, sentText: null };
}
