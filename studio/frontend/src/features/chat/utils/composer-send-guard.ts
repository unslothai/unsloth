// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Refuses writes that put a just-sent message back into the composer: an event the engine
 *  queued against the pre-send value (autocorrect commit, IME finalise) or a draft autosave
 *  that raced the send. Retired by user intent, never by a clock: queue latency has no upper
 *  bound, so a backgrounded page can deliver a stale write long after any window expired.
 *  Typing retires it since a typed value differs, and a paste retires it explicitly. */
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

/** Whether a keydown is the user starting to type again rather than part of the send. Enter is
 *  excluded because the send arms the guard from inside that very keydown, and a chord is
 *  excluded because it is a command, not a character. */
export function isGuardRetiringKey(event: {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey?: boolean;
  getModifierState?: (key: "AltGraph") => boolean;
}): boolean {
  // AltGr types characters but Windows reports it as Ctrl+Alt, so the chord check alone would
  // drop every character produced with it. Both forms are tested: some builds set the flags
  // even while AltGraph reads true.
  const altGraph =
    event.getModifierState?.("AltGraph") === true ||
    (event.ctrlKey && event.altKey === true && event.key.length === 1);
  if (!altGraph && (event.metaKey || event.ctrlKey)) return false;
  if (event.key === "Enter" || event.key === "Escape" || event.key === "Tab") {
    return false;
  }
  // Modifiers alone are not typing; the character keydown follows them.
  return !["Shift", "Control", "Alt", "Meta", "CapsLock"].includes(event.key);
}

/** A keydown, or a composition starting for dictation, handwriting or an IME. The send's queued
 *  writes are delivered before either, so one proves they have drained. Relaxes the equality
 *  check only, never the autocorrect rule or the draft suppression. */
export function markSentTextGuardUserInput(
  guard: SentTextGuard | null,
): SentTextGuard | null {
  if (guard === null || guard.userInputSince) return guard;
  return { ...guard, userInputSince: true };
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
    /** An autocorrect commit, the engine rather than the user. It cannot start from an empty
     *  composer, so on one it is stale, mutated value and all. */
    replacesText: boolean;
    /** Undo, redo, or text brought in from elsewhere. A queued write never reports one, so it
     *  applies even when it restores what was sent. */
    isDeliberate: boolean;
    /** An IME composition write. Stale only when the composition began before the send; one begun
     *  after raises compositionstart, which records user input. */
    isComposition: boolean;
    composerIsEmpty: boolean;
  },
): { accept: boolean; guard: SentTextGuard | null } {
  if (guard === null) return { accept: true, guard: null };
  if (write.isDeliberate) return { accept: true, guard: null };
  // Re-typing the whole prompt is only one write when it is one character, so equality alone
  // would swallow every retry of a "?" or a single emoji.
  if (guard.texts.includes(write.value)) {
    if (guard.userInputSince) return { accept: true, guard: null };
    return { accept: false, guard };
  }
  if (write.replacesText && write.composerIsEmpty) {
    return { accept: false, guard };
  }
  if (write.isComposition && write.composerIsEmpty && !guard.userInputSince) {
    return { accept: false, guard };
  }
  // Anything else is the user, so stop guarding rather than judge later writes.
  return { accept: true, guard: null };
}
