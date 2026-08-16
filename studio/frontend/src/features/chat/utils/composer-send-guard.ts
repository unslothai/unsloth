// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Refuses writes that put a just-sent message back into the composer.
 *
 * Sending clears the composer, but a write carrying the pre-send text can still
 * land after it: an input or compositionend event the engine queued against the
 * old value (autocorrect commit, IME finalise, undo), or a draft autosave that
 * raced the send. Applying it refills the composer and the next autosave makes
 * that stick, so the prompt sends and the text stays in the box.
 *
 * The guard is retired by user intent, not by a clock. Event queue latency has
 * no upper bound: a backgrounded page or a blocked main thread can deliver a
 * stale write seconds later, and any wall-clock window would have expired by
 * then. Typing retires it, since a typed value differs; a paste retires it
 * explicitly, since a deliberate re-paste of the same prompt is the one
 * legitimate write that is byte-identical to what was sent.
 */
export type SentTextGuard = {
  /** Values to refuse. The overlay send arms the wrapper and what was on screen. */
  readonly texts: readonly string[];
  /** Draft key the send cleared, so another thread's identical draft still restores. */
  readonly draftKey: string | null;
};

export function armSentTextGuard(
  texts: readonly string[],
  draftKey: string | null,
): SentTextGuard {
  return { texts: texts.filter((text) => text.length > 0), draftKey };
}

/** Whether a draft restore is the sent text coming back under the same key. */
export function sentTextGuardBlocksDraft(
  guard: SentTextGuard | null,
  draft: string,
  draftKey: string | null,
): boolean {
  if (guard === null) return false;
  return guard.draftKey === draftKey && guard.texts.includes(draft);
}

export function applySentTextGuard(
  guard: SentTextGuard | null,
  write: {
    value: string;
    /**
     * The engine replaced text rather than the user inserting it: an autocorrect
     * commit. It cannot originate from an empty composer, so on one it is
     * necessarily stale, mutated value and all.
     */
    replacesText: boolean;
    /**
     * The user asked for the previous value back. Deliberate, like a paste, so
     * it retires the guard even though it restores exactly what was sent.
     */
    isUndo: boolean;
    composerIsEmpty: boolean;
  },
): { accept: boolean; guard: SentTextGuard | null } {
  if (guard === null) return { accept: true, guard: null };
  if (write.isUndo) return { accept: true, guard: null };
  if (guard.texts.includes(write.value)) return { accept: false, guard };
  if (write.replacesText && write.composerIsEmpty) {
    return { accept: false, guard };
  }
  // Anything else is the user, so stop guarding rather than judge later writes.
  return { accept: true, guard: null };
}
