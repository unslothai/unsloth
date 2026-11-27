// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  bindingFromEvent,
  formatBindingValue,
} from "../../settings/lib/keyboard-shortcuts.ts";

export type ZoomDirection = 1 | -1 | 0;

/** A user shortcut bound to one of these wins. */
export const ZOOM_CHORDS: Record<ZoomDirection, string> = {
  1: "Mod+Equal",
  [-1]: "Mod+Minus",
  0: "Mod+Digit0",
};

type ZoomKeyEvent = Pick<
  KeyboardEvent,
  "key" | "code" | "metaKey" | "ctrlKey" | "altKey" | "shiftKey"
>;

export function zoomDirectionForKey(
  event: ZoomKeyEvent,
  mac: boolean,
): ZoomDirection | null {
  const mod = mac
    ? event.metaKey && !event.ctrlKey
    : event.ctrlKey && !event.metaKey;
  if (!mod || event.altKey) return null;
  if (event.key === "+" || event.key === "=" || event.code === "NumpadAdd")
    return 1;
  if (event.key === "-" || event.key === "_" || event.code === "NumpadSubtract")
    return -1;
  if (event.key === "0" || event.code === "Digit0" || event.code === "Numpad0")
    return 0;
  if (event.code === "Equal") return 1;
  if (event.code === "Minus") return -1;
  return null;
}

const ZOOM_CODES = new Set([
  "Equal",
  "Minus",
  "Digit0",
  "NumpadAdd",
  "NumpadSubtract",
  "Numpad0",
]);

// Digits keep their shortcut: Swiss and Turkish type "+" with Shift on 1 and 4, where Windows and
// Linux put Ctrl+Shift+1-9 (the workspace chords).
function keepsShortcut(code: string): boolean {
  return ZOOM_CODES.has(code) || code.startsWith("Digit");
}

export function zoomTypedOffZoomKey(
  event: ZoomKeyEvent,
  code: string,
  mac: boolean,
): boolean {
  return !keepsShortcut(code) && zoomDirectionForKey(event, mac) !== null;
}

/**
 * Whether a user shortcut takes this press instead: one bound to the canonical chord, or to the
 * exact keys pressed (Mod+Shift+Equal, a keypad key). Not on punctuation: German and most of Europe
 * type "-" on the Slash key, and Mod+Slash there is a shortcut, not the zoom the user typed.
 */
export function zoomChordTaken(
  event: ZoomKeyEvent,
  direction: ZoomDirection,
  mac: boolean,
  owned: (value: string) => boolean,
): boolean {
  if (owned(ZOOM_CHORDS[direction])) return true;
  const pressed = bindingFromEvent(event, mac);
  return (
    pressed !== null &&
    keepsShortcut(pressed.code) &&
    owned(formatBindingValue(pressed))
  );
}
