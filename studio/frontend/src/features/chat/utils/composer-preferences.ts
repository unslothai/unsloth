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

/** Whether a keydown is owned by IME and must not reach submit handling. */
export function imeKeydownBlocksComposerSubmit(
  event: ComposerKeyEvent,
  wasInCompositionSession: boolean,
): boolean {
  const ime = event.isComposing === true || event.keyCode === 229;
  if (!ime) return false;
  // macOS built-in Pinyin (#12137) can mark idle Enter as composing even when
  // no candidate session is active. Modifier chords stay IME-owned.
  if (
    event.key === "Enter" &&
    !wasInCompositionSession &&
    !event.metaKey &&
    !event.ctrlKey
  ) {
    return false;
  }
  return true;
}

/** Normalize an idle-IME Enter for `composerSubmitIntent`. */
export function composerKeyEventForImeSubmit(
  event: ComposerKeyEvent,
): ComposerKeyEvent {
  return {
    ...event,
    isComposing: false,
    keyCode: event.keyCode === 229 ? 13 : event.keyCode,
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
