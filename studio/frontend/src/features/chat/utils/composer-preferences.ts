// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "mod-enter-multiline" is Enter until the draft has a line break, then "mod-enter".
export type ComposerSendShortcut = "enter" | "mod-enter-multiline" | "mod-enter";
export type ComposerFollowUpBehavior = "queue" | "steer";
export type ComposerSubmitIntent = "default" | "opposite";

export type ComposerKeyEvent = {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  repeat?: boolean;
  isComposing?: boolean;
  keyCode?: number;
};

// WebKit fires compositionend before the committing keydown (keyCode 229, WebKit bug 165004); ProseMirror's window.
const IME_COMMIT_KEYDOWN_MS = 500;

/** False only for a plain IME-marked Enter outside any composition, e.g. idle macOS Pinyin (#12137). */
export function imeKeydownBlocksComposerSubmit(
  event: ComposerKeyEvent,
  imeSessionOpen: boolean,
  msSinceCompositionEnd: number,
): boolean {
  return (
    event.key !== "Enter" ||
    event.metaKey ||
    event.ctrlKey ||
    imeSessionOpen ||
    msSinceCompositionEnd < IME_COMMIT_KEYDOWN_MS
  );
}

export type InputImeState = {
  open: boolean;
  endedAt: number;
  commitKeydownSeen: boolean;
};

export function newInputImeState(): InputImeState {
  return { open: false, endedAt: -Infinity, commitKeydownSeen: false };
}

export function resetInputIme(ime: InputImeState) {
  ime.open = false;
  ime.endedAt = -Infinity;
  ime.commitKeydownSeen = false;
}

export function inputImeHandlers(ime: InputImeState) {
  // compositionend can go missing (#5546); a focus change always ends the composition.
  const reset = () => resetInputIme(ime);
  return {
    onFocus: reset,
    onBlur: reset,
    onCompositionStart: () => {
      ime.open = true;
      ime.commitKeydownSeen = false;
    },
    onCompositionEnd: (event: { timeStamp: number }) => {
      ime.open = false;
      ime.endedAt = ime.commitKeydownSeen ? -Infinity : event.timeStamp;
      ime.commitKeydownSeen = false;
    },
  };
}

/** True when the keydown belongs to an IME; idle macOS Pinyin Enter (229, #12137) passes. */
export function imeOwnsInputKeydown(
  event: ComposerKeyEvent & {
    timeStamp: number;
    nativeEvent: { isComposing?: boolean };
  },
  ime: InputImeState,
  options?: { modifiedEnterSubmits?: boolean },
): boolean {
  if (event.key === "Enter" && (event.nativeEvent.isComposing || ime.open)) {
    ime.commitKeydownSeen = true;
  }
  const msSinceCompositionEnd = event.timeStamp - ime.endedAt;
  ime.endedAt = -Infinity;
  // A held candidate-confirming Enter can repeat after compositionend. It is still the same
  // physical press, not the separate Enter that submits the input.
  if (event.key === "Enter" && event.repeat) return true;
  if (event.nativeEvent.isComposing) return true;
  if (event.keyCode !== 229) {
    // Candidate-confirming Enter can arrive as keyCode 13 mid-composition; swallow it once.
    const confirmsCandidate = ime.open && event.key === "Enter";
    ime.open = false;
    return confirmsCandidate;
  }
  const candidate = options?.modifiedEnterSubmits
    ? { ...event, metaKey: false, ctrlKey: false }
    : event;
  return imeKeydownBlocksComposerSubmit(candidate, ime.open, msSinceCompositionEnd);
}

export function composerKeyEventForImeSubmit(
  event: ComposerKeyEvent,
): ComposerKeyEvent {
  return {
    key: event.key,
    metaKey: event.metaKey,
    ctrlKey: event.ctrlKey,
    shiftKey: event.shiftKey,
    altKey: event.altKey,
    repeat: event.repeat,
  };
}

/** The rule in force for `draft`. */
export function effectiveSendShortcut(
  shortcut: ComposerSendShortcut,
  draft?: string | null,
): "enter" | "mod-enter" {
  if (shortcut !== "mod-enter-multiline") return shortcut;
  return draft?.includes("\n") ? "mod-enter" : "enter";
}

/** Only called for the focused composer, after its IME and mention-picker guards. */
export function composerSubmitIntent(
  event: ComposerKeyEvent,
  preference: ComposerSendShortcut,
  draft?: string | null,
): ComposerSubmitIntent | null {
  const shortcut = effectiveSendShortcut(preference, draft);
  if (
    event.key !== "Enter" ||
    event.altKey ||
    event.repeat ||
    event.isComposing ||
    event.keyCode === 229 ||
    (event.metaKey && event.ctrlKey)
  )
    return null;
  const mod = event.metaKey || event.ctrlKey;
  if (shortcut === "mod-enter") {
    return mod ? (event.shiftKey ? "opposite" : "default") : null;
  }
  if (event.shiftKey) return null;
  return mod ? "opposite" : "default";
}

export function composerFollowUpBehavior(
  preference: ComposerFollowUpBehavior,
  intent: ComposerSubmitIntent,
): ComposerFollowUpBehavior {
  if (intent === "default") return preference;
  return preference === "queue" ? "steer" : "queue";
}

/** Intent that lands a submit on `behavior` from either preference. */
export function followUpSubmitIntent(
  preference: ComposerFollowUpBehavior,
  behavior: ComposerFollowUpBehavior,
): ComposerSubmitIntent {
  return preference === behavior ? "default" : "opposite";
}

export function composerShortcutLabels(
  preference: ComposerSendShortcut,
  mac: boolean,
  draft?: string | null,
) {
  const shortcut = effectiveSendShortcut(preference, draft);
  const mod = mac ? "⌘" : "Ctrl+";
  return {
    send: shortcut === "enter" ? "Enter" : `${mod}Enter`,
    opposite:
      shortcut === "enter"
        ? `${mod}Enter`
        : mac
          ? "⇧⌘Enter"
          : "Ctrl+Shift+Enter",
  };
}

export function normalizeComposerPreferences(value: unknown) {
  const saved = value as Record<string, unknown> | null | undefined;
  return {
    plainTextComposer:
      typeof saved?.plainTextComposer === "boolean"
        ? saved.plainTextComposer
        : true,
    showContextWindowUsage:
      typeof saved?.showContextWindowUsage === "boolean"
        ? saved.showContextWindowUsage
        : true,
    sendShortcut: (saved?.sendShortcut === "mod-enter" ||
    saved?.sendShortcut === "mod-enter-multiline"
      ? saved.sendShortcut
      : "enter") as ComposerSendShortcut,
    followUpBehavior:
      saved?.followUpBehavior === "steer"
        ? ("steer" as const)
        : ("queue" as const),
  };
}

/** Put a steering prompt after a dispatched item, or before the next pending item. */
export function steeringInsertionIndex(
  items: readonly { dispatched: boolean }[],
  runIndex: number,
) {
  const index = Math.max(0, runIndex);
  return Math.min(items.length, index + (items[index]?.dispatched ? 1 : 0));
}
