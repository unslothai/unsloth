// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ComposerSendShortcut = "enter" | "mod-enter";
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

/** Only called for the focused composer, after its IME and mention-picker guards. */
export function composerSubmitIntent(
  event: ComposerKeyEvent,
  shortcut: ComposerSendShortcut,
): ComposerSubmitIntent | null {
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
  shortcut: ComposerSendShortcut,
  mac: boolean,
) {
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
    sendShortcut:
      saved?.sendShortcut === "mod-enter"
        ? ("mod-enter" as const)
        : ("enter" as const),
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
