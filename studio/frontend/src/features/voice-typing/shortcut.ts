// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Canonical chord strings ("Ctrl+Shift+Space") are the wire format between this
// UI and the native shortcut layer, which parses the same vocabulary. Everything
// here is pure so the settings capture box and the tests share one definition.

export const DEFAULT_VOICE_TYPING_SHORTCUT = "Ctrl+Shift+Space";

const MODIFIERS = ["Ctrl", "Shift", "Alt", "Super"] as const;
export type ShortcutModifier = (typeof MODIFIERS)[number];

/** KeyboardEvent.code -> canonical key label, for the keys the native layer knows. */
const CODE_TO_KEY: Record<string, string> = {
  Space: "Space",
  Enter: "Enter",
  NumpadEnter: "Enter",
  Tab: "Tab",
  Backspace: "Backspace",
  Minus: "-",
  Equal: "=",
  BracketLeft: "[",
  BracketRight: "]",
  Semicolon: ";",
  Quote: "'",
  Comma: ",",
  Period: ".",
  Slash: "/",
  Backslash: "\\",
  Backquote: "`",
};

const KEY_LABELS = new Set(Object.values(CODE_TO_KEY));

const MODIFIER_CODES = new Set([
  "ControlLeft",
  "ControlRight",
  "ShiftLeft",
  "ShiftRight",
  "AltLeft",
  "AltRight",
  "MetaLeft",
  "MetaRight",
]);

/** True while the user is still holding only modifiers and no real key yet. */
export function isModifierCode(code: string): boolean {
  return MODIFIER_CODES.has(code);
}

/** The subset of KeyboardEvent the capture needs, so tests can feed plain objects. */
export interface ChordEventLike {
  code: string;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  metaKey: boolean;
}

function letterFromCode(code: string): string | null {
  const match = /^Key([A-Z])$/.exec(code);
  return match ? match[1] : null;
}

function digitFromCode(code: string): string | null {
  const match = /^Digit([0-9])$/.exec(code);
  return match ? match[1] : null;
}

function functionFromCode(code: string): string | null {
  const match = /^F([1-9]|1[0-9]|2[0-4])$/.exec(code);
  return match ? `F${match[1]}` : null;
}

/** Canonical key label for a KeyboardEvent.code, or null when it cannot arm a chord. */
export function keyFromCode(code: string): string | null {
  const mapped = CODE_TO_KEY[code];
  if (mapped) return mapped;
  return letterFromCode(code) ?? digitFromCode(code) ?? functionFromCode(code);
}

/**
 * Builds a canonical chord from a key event. Null while the user is still pressing
 * modifiers (or pressed only unknown keys / a bare key without any modifier).
 */
export function chordFromEvent(event: ChordEventLike): string | null {
  // Still decking the modifiers: wait for the real key before judging.
  if (MODIFIER_CODES.has(event.code)) return null;
  const key = keyFromCode(event.code);
  if (!key) return null;
  const parts: string[] = [];
  if (event.ctrlKey) parts.push("Ctrl");
  if (event.shiftKey) parts.push("Shift");
  if (event.altKey) parts.push("Alt");
  if (event.metaKey) parts.push("Super");
  // A bare key or a modifier-only chord would shadow typing everywhere.
  if (parts.length === 0) return null;
  parts.push(key);
  return parts.join("+");
}

export function shortcutParts(shortcut: string): string[] {
  return shortcut
    .split("+")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function isModifierPart(part: string): part is ShortcutModifier {
  return (MODIFIERS as readonly string[]).includes(part);
}

/** True for strings the native parser accepts: at least one modifier and one key. */
export function isValidShortcut(shortcut: string): boolean {
  const parts = shortcutParts(shortcut);
  const modifiers = parts.filter(isModifierPart);
  const keys = parts.filter((part) => !isModifierPart(part));
  if (modifiers.length === 0 || keys.length !== 1) return false;
  const key = keys[0];
  return (
    /^[A-Z]$/.test(key) ||
    /^[0-9]$/.test(key) ||
    /^F([1-9]|1[0-9]|2[0-4])$/.test(key) ||
    KEY_LABELS.has(key)
  );
}

/** Human-readable form for the settings button ("Ctrl + Shift + Space"). */
export function formatShortcut(shortcut: string): string {
  return shortcutParts(shortcut)
    .map((part) => (part === "Super" ? "Win" : part))
    .join(" + ");
}
