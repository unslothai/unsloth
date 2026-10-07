// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

/**
 * To add one: SHORTCUT_DEFS entry, two i18n keys in all twelve locales, a label in settings-search.ts,
 * and a `useShortcut(id, ...)`. Order is render order and decides who owns a contested chord.
 */
export type ShortcutId =
  | "newChat"
  | "newTemporaryChat"
  | "archiveChat"
  | "newStandaloneChat"
  | "markChatUnread"
  | "togglePinChat"
  | "selectAllChats"
  | "clearChatSelection"
  | "deleteSelectedChats"
  | "nextRecentlyViewedChat"
  | "nextChat"
  | "nextChatNeedingAttention"
  | "previousRecentlyViewedChat"
  | "previousChat"
  | "goToRecentChat1"
  | "goToRecentChat2"
  | "goToRecentChat3"
  | "goToRecentChat4"
  | "goToRecentChat5"
  | "goToRecentChat6"
  | "switchToChat"
  | "switchToProjects"
  | "switchToHub"
  | "switchToTrain"
  | "switchToRecipes"
  | "switchToImages"
  | "switchToVideo"
  | "switchToAudio"
  | "switchToExport"
  | "findInPage"
  | "newBrowserTab"
  | "toggleBrowserFullView"
  | "toggleApiMonitor"
  | "toggleSidebar"
  | "openMcpServers"
  | "clearAllUnreads"
  | "logOut"
  | "openSettings"
  | "approveToolRequest"
  | "declineToolRequest"
  | "attachFiles"
  | "cycleReasoningEffort"
  | "decreaseReasoningEffort"
  | "increaseReasoningEffort"
  | "openModelPicker"
  | "openProjectPicker"
  | "startDictation"
  | "sendMessage"
  | "queueMessage"
  | "steerMessage"
  | "toggleFastMode"
  | "copyChatAsMarkdown"
  | "copySessionId"
  | "forkChat"
  | "openCommandPalette"
  | "searchChats"
  | "renameChat"
  | "openKeyboardShortcuts";

export type ShortcutSlot = "primary" | "alternate";

export const SHORTCUT_SLOTS: ShortcutSlot[] = ["primary", "alternate"];

export interface ShortcutDef {
  id: ShortcutId;
  labelKey: TranslationKey;
  descriptionKey: TranslationKey;
  defaultBinding: string | null;
  /** Second chord for the same action; both fire. */
  defaultAlternateBinding?: string | null;
  /** Off macOS Ctrl is Mod, so a ⌃ default would be unreachable there. */
  nonMacDefaultBinding?: string | null;
  nonMacDefaultAlternateBinding?: string | null;
  /** Only for prompt-gated actions. */
  allowBareKey?: boolean;
  /** The handler returns on desktop, so the row would offer a dead key. */
  webOnly?: boolean;
}

/** `code` is KeyboardEvent.code so bindings survive layout changes; `mod` is Cmd or Ctrl. */
export interface ShortcutBinding {
  code: string;
  mod: boolean;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
}

function def(
  id: ShortcutId,
  defaultBinding: string | null,
  extra: Omit<
    ShortcutDef,
    "id" | "labelKey" | "descriptionKey" | "defaultBinding"
  > = {},
): ShortcutDef {
  return {
    id,
    labelKey:
      `settings.keyboardShortcuts.actions.${id}.label` as TranslationKey,
    descriptionKey:
      `settings.keyboardShortcuts.actions.${id}.description` as TranslationKey,
    defaultBinding,
    ...extra,
  };
}

/** ⌘1-9 would read better but is browser tab switching. */
const RECENT_SLOT_DEFS: ShortcutDef[] = Array.from({ length: 6 }, (_, i) =>
  def(`goToRecentChat${i + 1}` as ShortcutId, `Mod+Alt+Digit${i + 1}`),
);

const WORKSPACE_IDS = [
  "switchToChat",
  "switchToProjects",
  "switchToHub",
  "switchToTrain",
  "switchToRecipes",
  "switchToImages",
  "switchToVideo",
  "switchToAudio",
  "switchToExport",
] as const;

/** Off macOS Ctrl+1-9 is browser tab switching, so Shift joins it. */
const WORKSPACE_DEFS: ShortcutDef[] = WORKSPACE_IDS.map((id, i) =>
  def(id, `Ctrl+Digit${i + 1}`, {
    nonMacDefaultBinding: `Mod+Shift+Digit${i + 1}`,
  }),
);

/** Order settles which action owns a chord two of them claim. */
export const SHORTCUT_DEFS: ShortcutDef[] = [
  // ⌘N opens a browser window uncancellably and ⇧⌘O already shipped, so ⌘N is the desktop alternate.
  def("newChat", "Mod+Shift+KeyO", { defaultAlternateBinding: "Mod+KeyN" }),
  def("newTemporaryChat", "Mod+Shift+KeyN"),
  // Not ⇧⌘A (Chrome tab search, Firefox add-ons); E is the mail archive key.
  def("archiveChat", "Mod+Alt+KeyE"),
  def("newStandaloneChat", "Mod+Alt+KeyO"),
  // ⌃⇧U off macOS is GTK/IBus hex entry.
  def("markChatUnread", "Mod+Shift+KeyU", {
    nonMacDefaultBinding: "Mod+Alt+KeyU",
  }),
  // ⌥⌘P is Chrome's Page Setup on macOS.
  def("togglePinChat", "Ctrl+Shift+KeyP", {
    nonMacDefaultBinding: "Mod+Alt+KeyP",
  }),
  def("selectAllChats", "Mod+Alt+KeyS"),
  // Escape clears it from the sidebar's own listener; bare Escape here declines a tool call.
  def("clearChatSelection", null),
  // Unassigned on purpose: nothing that deletes chats should ship on a chord.
  def("deleteSelectedChats", null),
  def("nextRecentlyViewedChat", "Ctrl+Tab", {
    nonMacDefaultBinding: "Mod+Tab",
  }),
  // No arrow alternate: ⌥⌘→ is Chrome's next tab, and Ctrl+Alt+→ is desktop switching off macOS.
  def("nextChat", "Mod+Shift+BracketRight"),
  def("nextChatNeedingAttention", "Mod+Alt+KeyA"),
  def("previousRecentlyViewedChat", "Ctrl+Shift+Tab", {
    nonMacDefaultBinding: "Mod+Shift+Tab",
  }),
  def("previousChat", "Mod+Shift+BracketLeft"),
  ...RECENT_SLOT_DEFS,

  ...WORKSPACE_DEFS,

  // Deliberately takes the browser's find, which cannot reach unmounted messages and is absent on
  // desktop; cancellable everywhere, and Settings still flags it.
  def("findInPage", "Mod+KeyF"),
  // Bookmarks in Chrome and Firefox, but cancellable.
  def("newBrowserTab", "Mod+Shift+KeyB"),
  def("toggleBrowserFullView", "Mod+Shift+KeyF"),
  // ⌥⌘U is view source on macOS; off macOS U is taken by the two unread actions, so M for monitor.
  def("toggleApiMonitor", "Ctrl+Shift+KeyU", {
    nonMacDefaultBinding: "Mod+Alt+Shift+KeyM",
  }),
  def("toggleSidebar", "Mod+KeyB"),
  def("openMcpServers", null),
  // ⇧Esc is Chrome's and Edge's task manager off macOS.
  def("clearAllUnreads", "Shift+Escape", {
    nonMacDefaultBinding: "Mod+Alt+Shift+KeyU",
  }),
  // Desktop signs out through the OS account menu, so this chord could not fire there.
  def("logOut", null, { webOnly: true }),
  def("openSettings", "Mod+Comma"),
  // Bare ⏎ / Esc register only while a tool call is waiting.
  def("approveToolRequest", "Enter", { allowBareKey: true }),
  def("declineToolRequest", "Escape", { allowBareKey: true }),

  def("attachFiles", null),
  def("cycleReasoningEffort", null),
  def("decreaseReasoningEffort", null),
  def("increaseReasoningEffort", null),
  // Off macOS ⇧ collides with Chrome's profile switcher and bookmark-all-tabs.
  def("openModelPicker", "Ctrl+Shift+KeyM", {
    nonMacDefaultBinding: "Mod+Alt+KeyM",
  }),
  def("openProjectPicker", "Mod+Alt+Shift+KeyO"),
  def("startDictation", "Ctrl+Shift+KeyD", {
    nonMacDefaultBinding: "Mod+Alt+KeyV",
  }),
  def("sendMessage", null),
  // Unassigned: ⌘⏎ already sends with the opposite follow-up.
  def("queueMessage", null),
  def("steerMessage", null),
  def("toggleFastMode", null),

  def("copyChatAsMarkdown", null),
  // ⌥⌘C is Firefox's element picker and Safari's console.
  def("copySessionId", "Ctrl+Shift+KeyC", {
    nonMacDefaultBinding: "Mod+Alt+KeyC",
  }),
  def("forkChat", null),
  // Takes the browser's Print, which is useless on a chat shell and cancellable.
  def("openCommandPalette", "Mod+KeyP"),
  // No ⇧⌘P alternate: it opens a private window in Firefox.
  def("searchChats", "Mod+KeyK"),
  def("renameChat", "Mod+Alt+KeyR"),
  def("openKeyboardShortcuts", "Mod+Slash"),
];

export const SHORTCUT_DEF_BY_ID: Record<ShortcutId, ShortcutDef> =
  Object.fromEntries(SHORTCUT_DEFS.map((d) => [d.id, d])) as Record<
    ShortcutId,
    ShortcutDef
  >;

const SHORTCUT_IDS = new Set<string>(SHORTCUT_DEFS.map((d) => d.id));

export function isShortcutId(value: unknown): value is ShortcutId {
  return typeof value === "string" && SHORTCUT_IDS.has(value);
}

export function isShortcutSlot(value: unknown): value is ShortcutSlot {
  return value === "primary" || value === "alternate";
}

export function defaultBindingFor(
  def: ShortcutDef,
  slot: ShortcutSlot,
  mac = isMacPlatform(),
): string | null {
  if (slot === "alternate") {
    if (!mac && def.nonMacDefaultAlternateBinding !== undefined) {
      return def.nonMacDefaultAlternateBinding;
    }
    return def.defaultAlternateBinding ?? null;
  }
  if (!mac && def.nonMacDefaultBinding !== undefined) {
    return def.nonMacDefaultBinding;
  }
  return def.defaultBinding;
}

/**
 * Chords every browser user loses out of the box; both the warning and a test rely on them being
 * truly gone, so chords behind hidden developer menus (Safari's ⌥⌘E) do not qualify.
 */
const BROWSER_RESERVED_VALUES = new Set<string>([
  "Mod+KeyN",
  "Mod+Shift+KeyN",
  "Mod+KeyT",
  "Mod+Shift+KeyT",
  "Mod+KeyW",
  "Mod+Shift+KeyW",
  "Mod+KeyL",
  // Unsloth ships its own find on it anyway; this warns before a web user rebinds onto it.
  "Mod+KeyF",
  "Mod+KeyR",
  "Mod+Shift+KeyR",
  "Mod+KeyP",
  // Firefox's private window; Chrome's ⇧⌘N is already listed.
  "Mod+Shift+KeyP",
  "Mod+Shift+KeyA",
  "Mod+Tab",
  "Mod+Shift+Tab",
  "Ctrl+Tab",
  "Ctrl+Shift+Tab",
  ...Array.from({ length: 9 }, (_, i) => `Mod+Digit${i + 1}`),
]);

/** ⌥⌘ is the browsers' own run on macOS; off macOS these read as unclaimed Ctrl+Alt. */
const MAC_RESERVED_VALUES = new Set<string>([
  "Mod+Alt+KeyU",
  "Mod+Alt+KeyI",
  "Mod+Alt+KeyJ",
  "Mod+Alt+KeyB",
  "Mod+Alt+KeyN",
  "Mod+Alt+KeyF",
  "Mod+Alt+KeyC",
  "Mod+Alt+KeyK",
  "Mod+Alt+KeyP",
  // Safari and Chrome next/previous tab on macOS; the desktop chat walk uses them.
  "Mod+Shift+BracketLeft",
  "Mod+Shift+BracketRight",
  "Mod+Alt+ArrowLeft",
  "Mod+Alt+ArrowRight",
  "Mod+Alt+ArrowUp",
  "Mod+Alt+ArrowDown",
]);

const NON_MAC_RESERVED_VALUES = new Set<string>([
  "Shift+Escape",
  "Mod+PageUp",
  "Mod+PageDown",
  "Alt+ArrowLeft",
  "Alt+ArrowRight",
  ...Array.from({ length: 9 }, (_, i) => `Alt+Digit${i + 1}`),
]);

export function isBrowserReservedBinding(
  value: string | null,
  mac = isMacPlatform(),
): boolean {
  if (value === null) return false;
  if (BROWSER_RESERVED_VALUES.has(value)) return true;
  return mac
    ? MAC_RESERVED_VALUES.has(value)
    : NON_MAC_RESERVED_VALUES.has(value);
}

/** A bare one must not cancel a focused button's own activation via preventDefault. */
const ACTIVATION_CODES = new Set(["Enter", "NumpadEnter", "Space"]);

const ACTIVATABLE_TAGS = new Set(["BUTTON", "A", "SUMMARY", "SELECT", "OPTION"]);

export function activationBelongsToFocus(
  binding: ShortcutBinding,
  el: { tagName?: string; getAttribute?: (name: string) => string | null } | null,
): boolean {
  if (binding.mod || binding.ctrl || binding.alt || binding.shift) return false;
  if (!ACTIVATION_CODES.has(binding.code)) return false;
  if (!el) return false;
  if (ACTIVATABLE_TAGS.has(el.tagName ?? "")) return true;
  const role = el.getAttribute?.("role") ?? null;
  return role === "button" || role === "link" || role === "menuitem";
}

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

// Resolved once: matchesBinding runs on every keydown.
let macPlatform: boolean | null = null;

export function isMacPlatform(): boolean {
  if (macPlatform !== null) return macPlatform;
  if (typeof navigator === "undefined") return false;
  // `platform` is deprecated but still the only synchronous signal in Safari.
  const source = `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`;
  macPlatform = /mac|iphone|ipad|ipod/i.test(source);
  return macPlatform;
}

export function formatBindingValue(binding: ShortcutBinding): string {
  const parts: string[] = [];
  if (binding.mod) parts.push("Mod");
  if (binding.ctrl) parts.push("Ctrl");
  if (binding.alt) parts.push("Alt");
  if (binding.shift) parts.push("Shift");
  parts.push(binding.code);
  return parts.join("+");
}

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

export function bindingFromEvent(
  event: {
    code: string;
    key?: string;
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    getModifierState?: (key: string) => boolean;
  },
  mac = isMacPlatform(),
): ShortcutBinding | null {
  const code = event.code || keyToCode(event.key ?? "");
  if (!code || isModifierCode(code)) return null;
  // Off macOS Meta cannot be stored, so recording Super+Alt+K would save plain Alt+K; record nothing.
  if (!mac && event.metaKey) return null;
  // matchesBinding will not fire an AltGr chord, so do not record one.
  if (!mac && isAltGraphEvent(event)) return null;
  // Cmd on macOS and Ctrl elsewhere record as Mod; a macOS user pressing Ctrl means Ctrl.
  return {
    code,
    mod: mac ? event.metaKey : event.ctrlKey,
    ctrl: mac ? event.ctrlKey : false,
    shift: event.shiftKey,
    alt: event.altKey,
  };
}

/** AltGr reports Ctrl+Alt, so typing ą or € would fire chords. Off macOS only: Option reports
 * AltGraph too. */
function isAltGraphEvent(event: {
  getModifierState?: (key: string) => boolean;
}): boolean {
  return event.getModifierState?.("AltGraph") === true;
}

/** Fallback for engines that report an empty `code` (some IMEs). */
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
    getModifierState?: (key: string) => boolean;
  },
  mac = isMacPlatform(),
): boolean {
  const code = event.code || keyToCode(event.key ?? "");
  if (code !== binding.code) return false;
  // Off macOS a Ctrl chord is unreachable; without this a Mac value would fire on the bare key.
  if (!mac && binding.ctrl) return false;
  if (!mac && binding.alt && isAltGraphEvent(event)) return false;
  const modHeld = mac ? event.metaKey : event.ctrlKey;
  const otherModHeld = mac ? event.ctrlKey : event.metaKey;
  if (modHeld !== binding.mod) return false;
  if (mac) {
    if (event.ctrlKey !== binding.ctrl) return false;
  } else if (otherModHeld) {
    return false;
  }
  return event.shiftKey === binding.shift && event.altKey === binding.alt;
}

/** Subset match for searching by chord; matchesBinding is exact. */
export function keystrokeMatchesBinding(
  pressed: ShortcutBinding,
  bound: ShortcutBinding,
): boolean {
  if (pressed.code !== bound.code) return false;
  if (pressed.mod && !bound.mod) return false;
  if (pressed.ctrl && !bound.ctrl) return false;
  if (pressed.shift && !bound.shift) return false;
  if (pressed.alt && !bound.alt) return false;
  return true;
}

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

export function formatBindingValueLabel(
  value: string | null,
  mac = isMacPlatform(),
): string | null {
  const parsed = parseBinding(value);
  return parsed ? formatBindingLabel(parsed, mac) : null;
}

/** A bare key would swallow typing; function keys and `allowBareKey` actions are exempt. */
export function isAcceptableBinding(
  binding: ShortcutBinding,
  allowBareKey = false,
): boolean {
  if (binding.mod || binding.ctrl || binding.alt) return true;
  // Bare Tab would make the prompt's own buttons unreachable by keyboard.
  if (binding.code === "Tab") return false;
  if (allowBareKey) return true;
  if (/^F\d{1,2}$/.test(binding.code)) return true;
  // Bare Escape declines tool calls and exits the recorder; with Shift it is free.
  return binding.code === "Escape" && binding.shift;
}
