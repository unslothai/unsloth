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
 */

/** Stale writes land within a frame or two; a re-paste this fast is not real. */
export const SENT_TEXT_GUARD_MS = 500;

export type SentTextGuard = {
  /** Values to refuse. The overlay send arms both the wrapper and what was on screen. */
  readonly texts: readonly string[];
  /** Draft key the send cleared, so another thread's identical draft still restores. */
  readonly draftKey: string | null;
  readonly expiresAt: number;
};

export function armSentTextGuard(
  texts: readonly string[],
  draftKey: string | null,
  now: number,
): SentTextGuard {
  return {
    texts: texts.filter((text) => text.length > 0),
    draftKey,
    expiresAt: now + SENT_TEXT_GUARD_MS,
  };
}

/** Still inside the window a stale write can arrive in. */
export function sentTextGuardLive(
  guard: SentTextGuard | null,
  now: number,
): guard is SentTextGuard {
  return guard !== null && now < guard.expiresAt;
}

/** Whether a draft restore is the sent text coming back under the same key. */
export function sentTextGuardBlocksDraft(
  guard: SentTextGuard | null,
  draft: string,
  draftKey: string | null,
  now: number,
): boolean {
  if (!sentTextGuardLive(guard, now)) return false;
  return guard.draftKey === draftKey && guard.texts.includes(draft);
}

export function applySentTextGuard(
  guard: SentTextGuard | null,
  write: {
    value: string;
    /**
     * The engine replaced text rather than the user inserting it: an autocorrect
     * commit or an undo. Such a write cannot originate from an empty composer,
     * so on one it is necessarily stale, mutated value and all.
     */
    replacesText: boolean;
    composerIsEmpty: boolean;
  },
  now: number,
): { accept: boolean; guard: SentTextGuard | null } {
  // Expiry is what lets a deliberate identical re-paste through, and bounds the
  // guard to the window the stale writes actually arrive in.
  if (!sentTextGuardLive(guard, now)) return { accept: true, guard: null };
  if (guard.texts.includes(write.value)) return { accept: false, guard };
  if (write.replacesText && write.composerIsEmpty) {
    return { accept: false, guard };
  }
  // Anything else is the user, so stop guarding rather than judge later writes.
  return { accept: true, guard: null };
}
