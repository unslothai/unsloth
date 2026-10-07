// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Refuses stale writes that put a just-sent message back in the composer (IME, autocorrect,
 *  draft autosave). Retired by user input, never by a clock: queue latency is unbounded. */
export type SentTextGuard = {
  readonly texts: readonly string[];
  /** Draft key the send cleared, so another thread's identical draft still restores. */
  readonly draftKey: string | null;
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

/** Enter is excluded since the send arms the guard inside that keydown; chords are commands. */
export function isGuardRetiringKey(event: {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey?: boolean;
  getModifierState?: (key: "AltGraph") => boolean;
}): boolean {
  // Windows reports AltGr as Ctrl+Alt, so test both forms or every AltGr character is dropped.
  const altGraph =
    event.getModifierState?.("AltGraph") === true ||
    (event.ctrlKey && event.altKey === true && event.key.length === 1);
  if (!altGraph && (event.metaKey || event.ctrlKey)) return false;
  if (event.key === "Enter" || event.key === "Escape" || event.key === "Tab") {
    return false;
  }
  return !["Shift", "Control", "Alt", "Meta", "CapsLock"].includes(event.key);
}

/** The send's queued writes drain before a keydown or compositionstart, so either proves it. */
export function markSentTextGuardUserInput(
  guard: SentTextGuard | null,
): SentTextGuard | null {
  if (guard === null || guard.userInputSince) return guard;
  return { ...guard, userInputSince: true };
}

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
    /** Autocorrect cannot start from an empty composer, so on one it is stale. */
    replacesText: boolean;
    /** Undo, redo, or external text; a queued write never reports one. */
    isDeliberate: boolean;
    /** Stale only when the composition began before the send. */
    isComposition: boolean;
    composerIsEmpty: boolean;
  },
): { accept: boolean; guard: SentTextGuard | null } {
  if (guard === null) return { accept: true, guard: null };
  if (write.isDeliberate) return { accept: true, guard: null };
  // Equality alone would swallow every retry of a one-character prompt.
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
  return { accept: true, guard: null };
}
