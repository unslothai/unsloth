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
  /** A key has been pressed since the send. See markSentTextGuardUserInput. */
  readonly userInputSince: boolean;
};

export function armSentTextGuard(
  texts: readonly string[],
  draftKey: string | null,
): SentTextGuard {
  return {
    texts: texts.filter((text) => text.length > 0),
    draftKey,
    userInputSince: false,
  };
}

/**
 * Whether a keydown is the user starting to type again rather than part of the
 * send. Enter is excluded because the send arms the guard from inside that very
 * keydown, and a chord is excluded because it is a command, not a character.
 */
export function isGuardRetiringKey(event: {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
}): boolean {
  if (event.metaKey || event.ctrlKey) return false;
  if (event.key === "Enter" || event.key === "Escape" || event.key === "Tab") {
    return false;
  }
  // Modifiers alone are not typing; the character keydown follows them.
  return !["Shift", "Control", "Alt", "Meta", "CapsLock"].includes(event.key);
}

/**
 * Records that the user drove input after the send: a keydown, or a composition
 * starting for dictation, handwriting or an IME.
 *
 * A write the send queued is delivered before either of those, so one proves
 * the stale writes have already drained and this composer is the user's again.
 * Only the equality check relaxes: the autocorrect rule and the draft
 * suppression are unaffected, so this cannot widen into the original bug.
 */
export function markSentTextGuardUserInput(
  guard: SentTextGuard | null,
): SentTextGuard | null {
  if (guard === null || guard.userInputSince) return guard;
  return { ...guard, userInputSince: true };
}

/**
 * Whether a draft restore is the sent text coming back under the same key.
 *
 * A draft the composer already holds is not a raced save. Code outside the
 * composer fills it directly, the saved-prompt menu being the case in hand, and
 * that write never passes the guard, so the draft it autosaves would otherwise
 * be read as raced and deleted. A raced save is only ever a value the composer
 * does not have: the send cleared it, or another thread's text is on screen.
 */
export function sentTextGuardBlocksDraft(
  guard: SentTextGuard | null,
  draft: string,
  draftKey: string | null,
  composerText: string,
): boolean {
  if (guard === null) return false;
  if (composerText === draft) return false;
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
  // Re-typing the whole prompt is only one write when it is one character, so
  // equality alone would swallow every retry of a "?" or a single emoji.
  if (guard.texts.includes(write.value)) {
    if (guard.userInputSince) return { accept: true, guard: null };
    return { accept: false, guard };
  }
  if (write.replacesText && write.composerIsEmpty) {
    return { accept: false, guard };
  }
  // Anything else is the user, so stop guarding rather than judge later writes.
  return { accept: true, guard: null };
}
