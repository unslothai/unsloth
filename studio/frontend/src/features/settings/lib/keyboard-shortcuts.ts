// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

/**
 * Every rebindable action. To add one: an entry in SHORTCUT_DEFS, its two i18n
 * keys, and a `useShortcut(id, ...)` where the action runs.
 */
export type ShortcutId =
  | "openSettings"
  | "openKeyboardShortcuts"
  | "newChat"
  | "searchChats"
  | "toggleSidebar";

export type ShortcutGroup = "general" | "chat";

export interface ShortcutDef {
  id: ShortcutId;
  labelKey: TranslationKey;
  descriptionKey: TranslationKey;
  group: ShortcutGroup;
  /** Serialized binding, or null for an action that ships unassigned. */
  defaultBinding: string | null;
}

/**
 * `code` is KeyboardEvent.code, not `key`: the physical key, so a binding
 * survives a layout change and does not shift under Shift/Option. `mod` is Cmd
 * on macOS and Ctrl elsewhere, as every handler here already treated it.
 */
export interface ShortcutBinding {
  code: string;
  mod: boolean;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
}

export const SHORTCUT_GROUPS: ShortcutGroup[] = ["general", "chat"];

export const SHORTCUT_DEFS: ShortcutDef[] = [
  {
    id: "newChat",
    labelKey: "settings.keyboardShortcuts.actions.newChat.label",
    descriptionKey: "settings.keyboardShortcuts.actions.newChat.description",
    group: "chat",
    defaultBinding: "Mod+Shift+KeyO",
  },
  {
    id: "searchChats",
    labelKey: "settings.keyboardShortcuts.actions.searchChats.label",
    descriptionKey:
      "settings.keyboardShortcuts.actions.searchChats.description",
    group: "chat",
    defaultBinding: "Mod+KeyK",
  },
  {
    id: "toggleSidebar",
    labelKey: "settings.keyboardShortcuts.actions.toggleSidebar.label",
    descriptionKey:
      "settings.keyboardShortcuts.actions.toggleSidebar.description",
    group: "general",
    defaultBinding: "Mod+KeyB",
  },
  {
    id: "openSettings",
    labelKey: "settings.keyboardShortcuts.actions.openSettings.label",
    descriptionKey:
      "settings.keyboardShortcuts.actions.openSettings.description",
    group: "general",
    defaultBinding: "Mod+Comma",
  },
  {
    id: "openKeyboardShortcuts",
    labelKey: "settings.keyboardShortcuts.actions.openKeyboardShortcuts.label",
    descriptionKey:
      "settings.keyboardShortcuts.actions.openKeyboardShortcuts.description",
    group: "general",
    defaultBinding: "Mod+Slash",
  },
];

export const SHORTCUT_DEF_BY_ID: Record<ShortcutId, ShortcutDef> =
  Object.fromEntries(SHORTCUT_DEFS.map((def) => [def.id, def])) as Record<
    ShortcutId,
    ShortcutDef
  >;

const SHORTCUT_IDS = new Set<string>(SHORTCUT_DEFS.map((def) => def.id));

export function isShortcutId(value: unknown): value is ShortcutId {
  return typeof value === "string" && SHORTCUT_IDS.has(value);
}

/** Modifier codes are never a binding's key on their own. */
const MODIFIER_CODES = new Set([
  "MetaLeft",
  "MetaRight",
  "ControlLeft",
  "ControlRight",
  "ShiftLeft",
  "ShiftRight",
  "AltLeft",
  "AltRight",
  "CapsLock",
]);

export function isModifierCode(code: string): boolean {
  return MODIFIER_CODES.has(code);
}

// Resolved once: matchesBinding runs on every keydown, and the platform cannot
// change under a live document.
let macPlatform: boolean | null = null;

export function isMacPlatform(): boolean {
  if (macPlatform !== null) return macPlatform;
  if (typeof navigator === "undefined") return false;
  // `platform` is deprecated but still the only synchronous signal in Safari,
  // which is the browser most likely to be rendering ⌘ here.
  const source = `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`;
  macPlatform = /mac|iphone|ipad|ipod/i.test(source);
  return macPlatform;
}

/** Serialize to the stored form, e.g. "Mod+Shift+KeyO". */
export function formatBindingValue(binding: ShortcutBinding): string {
  const parts: string[] = [];
  if (binding.mod) parts.push("Mod");
  if (binding.ctrl) parts.push("Ctrl");
  if (binding.alt) parts.push("Alt");
  if (binding.shift) parts.push("Shift");
  parts.push(binding.code);
  return parts.join("+");
}

/** Parse the stored form. Returns null for anything unrecognised. */
export function parseBinding(
  value: string | null | undefined,
): ShortcutBinding | null {
  if (!value) return null;
  const parts = value.split("+").filter(Boolean);
  if (parts.length === 0) return null;
  const code = parts[parts.length - 1];
  if (!code || isModifierCode(code)) return null;
  const binding: ShortcutBinding = {
    code,
    mod: false,
    ctrl: false,
    shift: false,
    alt: false,
  };
  for (const part of parts.slice(0, -1)) {
    switch (part) {
      case "Mod":
        binding.mod = true;
        break;
      case "Ctrl":
        binding.ctrl = true;
        break;
      case "Shift":
        binding.shift = true;
        break;
      case "Alt":
        binding.alt = true;
        break;
      default:
        return null;
    }
  }
  return binding;
}

/** Build a binding from a keydown. Returns null while only modifiers are held. */
export function bindingFromEvent(
  event: {
    code: string;
    key?: string;
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
  },
  mac = isMacPlatform(),
): ShortcutBinding | null {
  const code = event.code || keyToCode(event.key ?? "");
  if (!code || isModifierCode(code)) return null;
  // Off macOS there is nowhere to put Meta: matchesBinding rejects an event
  // carrying it, so recording Super+Alt+K would drop the Super and persist plain
  // Alt+K -- a chord the user did not choose, which then fires on Alt+K alone
  // while the one they pressed never matches. Record nothing instead, the same
  // answer the recorder already gets while only modifiers are held.
  if (!mac && event.metaKey) return null;
  // Cmd on macOS and Ctrl elsewhere both record as Mod, so one binding reads
  // naturally on either platform. A macOS user pressing Ctrl means Ctrl.
  return {
    code,
    mod: mac ? event.metaKey : event.ctrlKey,
    ctrl: mac ? event.ctrlKey : false,
    shift: event.shiftKey,
    alt: event.altKey,
  };
}

/** Last-resort code for engines that report an empty `code` (some IMEs). */
function keyToCode(key: string): string {
  if (!key) return "";
  if (/^[a-z]$/i.test(key)) return `Key${key.toUpperCase()}`;
  if (/^[0-9]$/.test(key)) return `Digit${key}`;
  const punctuation: Record<string, string> = {
    ",": "Comma",
    ".": "Period",
    "/": "Slash",
    ";": "Semicolon",
    "'": "Quote",
    "[": "BracketLeft",
    "]": "BracketRight",
    "\\": "Backslash",
    "-": "Minus",
    "=": "Equal",
    "`": "Backquote",
  };
  return punctuation[key] ?? key;
}

export function matchesBinding(
  binding: ShortcutBinding,
  event: {
    code: string;
    key?: string;
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
  },
  mac = isMacPlatform(),
): boolean {
  const code = event.code || keyToCode(event.key ?? "");
  if (code !== binding.code) return false;
  const modHeld = mac ? event.metaKey : event.ctrlKey;
  // Off-platform modifier: on macOS a bare Ctrl must not satisfy a Mod binding,
  // and on Windows/Linux the Meta (Windows) key must not either.
  const otherModHeld = mac ? event.ctrlKey : event.metaKey;
  if (modHeld !== binding.mod) return false;
  if (mac) {
    if (event.ctrlKey !== binding.ctrl) return false;
  } else if (otherModHeld) {
    return false;
  }
  return event.shiftKey === binding.shift && event.altKey === binding.alt;
}

/** Human label for a code: "KeyO" -> "O", "Comma" -> ",", "ArrowUp" -> "↑". */
export function formatCode(code: string): string {
  if (code.startsWith("Key") && code.length === 4) return code.slice(3);
  if (code.startsWith("Digit") && code.length === 6) return code.slice(5);
  const named: Record<string, string> = {
    Comma: ",",
    Period: ".",
    Slash: "/",
    Semicolon: ";",
    Quote: "'",
    BracketLeft: "[",
    BracketRight: "]",
    Backslash: "\\",
    Minus: "-",
    Equal: "=",
    Backquote: "`",
    Space: "Space",
    Enter: "Enter",
    Escape: "Esc",
    Backspace: "⌫",
    Delete: "Del",
    Tab: "Tab",
    ArrowUp: "↑",
    ArrowDown: "↓",
    ArrowLeft: "←",
    ArrowRight: "→",
  };
  return named[code] ?? code;
}

/**
 * Display string, in the platform's own modifier order: macOS renders
 * ⌃⌥⇧⌘ then the key, everything else spells the modifiers out.
 */
export function formatBindingLabel(
  binding: ShortcutBinding,
  mac = isMacPlatform(),
): string {
  const key = formatCode(binding.code);
  if (mac) {
    let out = "";
    if (binding.ctrl) out += "⌃";
    if (binding.alt) out += "⌥";
    if (binding.shift) out += "⇧";
    if (binding.mod) out += "⌘";
    return `${out}${key}`;
  }
  const parts: string[] = [];
  if (binding.mod || binding.ctrl) parts.push("Ctrl");
  if (binding.alt) parts.push("Alt");
  if (binding.shift) parts.push("Shift");
  parts.push(key);
  return parts.join("+");
}

/** Convenience for rendering a stored value straight to a chip. */
export function formatBindingValueLabel(
  value: string | null,
  mac = isMacPlatform(),
): string | null {
  const parsed = parseBinding(value);
  return parsed ? formatBindingLabel(parsed, mac) : null;
}

/**
 * A binding with no modifier at all would swallow plain typing, so the
 * recorder refuses it. Function keys and Escape are self-contained.
 */
export function isAcceptableBinding(binding: ShortcutBinding): boolean {
  if (binding.mod || binding.ctrl || binding.alt) return true;
  return /^F\d{1,2}$/.test(binding.code) || binding.code === "Escape";
}
