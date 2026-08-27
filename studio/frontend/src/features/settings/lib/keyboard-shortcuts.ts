// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

/**
 * Every rebindable action. To add one: an entry in SHORTCUT_DEFS, its two i18n
 * keys in all twelve locales, its label in settings-search.ts, and a
 * `useShortcut(id, ...)` where the action runs.
 *
 * Order is the render order and decides who owns a contested chord, so the
 * most reached-for actions come first. An action only earns a row when Unsloth
 * has the feature behind it.
 */
export type ShortcutId =
  // Chat navigation, first because it is what the list is mostly for.
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
  // Workspaces
  | "switchToChat"
  | "switchToProjects"
  | "switchToHub"
  | "switchToTrain"
  | "switchToRecipes"
  | "switchToImages"
  | "switchToVideo"
  | "switchToAudio"
  | "switchToExport"
  // Panels and app-level actions
  | "toggleApiMonitor"
  | "toggleSidebar"
  | "openMcpServers"
  | "clearAllUnreads"
  | "logOut"
  | "openSettings"
  | "approveToolRequest"
  | "declineToolRequest"
  // Composer
  | "attachFiles"
  | "cycleReasoningEffort"
  | "decreaseReasoningEffort"
  | "increaseReasoningEffort"
  | "openModelPicker"
  | "openProjectPicker"
  | "startDictation"
  | "sendMessage"
  | "toggleFastMode"
  // Chat actions
  | "copyChatAsMarkdown"
  | "copySessionId"
  | "forkChat"
  | "searchChats"
  | "renameChat"
  | "openKeyboardShortcuts";

/** Which of an action's two chords a value belongs to. */
export type ShortcutSlot = "primary" | "alternate";

export const SHORTCUT_SLOTS: ShortcutSlot[] = ["primary", "alternate"];

export interface ShortcutDef {
  id: ShortcutId;
  labelKey: TranslationKey;
  descriptionKey: TranslationKey;
  /** Serialized binding, or null for an action that ships unassigned. */
  defaultBinding: string | null;
  /** Second chord for the same action, e.g. ⌥⌘→ beside ⇧⌘]. Both fire. */
  defaultAlternateBinding?: string | null;
  /** Off macOS Ctrl is Mod, so a ⌃ default would be unreachable there. */
  nonMacDefaultBinding?: string | null;
  nonMacDefaultAlternateBinding?: string | null;
  /** Allow a chord with no modifier. Only for prompt-gated actions. */
  allowBareKey?: boolean;
  /**
   * Hide the row on the desktop build. The handler returns there, so offering
   * the row offers a key that does nothing.
   */
  webOnly?: boolean;
}

/** `code` is KeyboardEvent.code, so a binding survives a layout change. `mod`
 *  is Cmd on macOS and Ctrl elsewhere. */
export interface ShortcutBinding {
  code: string;
  mod: boolean;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
}

/** Short-hand for the many rows whose two i18n keys follow the id. */
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

/** Rows 1-6 of Recents. ⌘1-9 would read better but is browser tab switching. */
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

/** ⌃1-9 on macOS. Off macOS that run is Ctrl+1-9, browser tab switching, so
 *  Shift joins it. */
const WORKSPACE_DEFS: ShortcutDef[] = WORKSPACE_IDS.map((id, i) =>
  def(id, `Ctrl+Digit${i + 1}`, {
    nonMacDefaultBinding: `Mod+Shift+Digit${i + 1}`,
  }),
);

/** Most-used first: the list renders in this order, and it settles which
 *  action owns a chord two of them claim, rather than mount order. */
export const SHORTCUT_DEFS: ShortcutDef[] = [
  // -- Chat navigation ----------------------------------------------------
  // ⌘N opens a browser window and cannot be prevented, and ⇧⌘O already
  // shipped, so ⌘N rides along as the alternate for the desktop build.
  def("newChat", "Mod+Shift+KeyO", { defaultAlternateBinding: "Mod+KeyN" }),
  def("newTemporaryChat", "Mod+Shift+KeyN"),
  // ⇧⌘A is Chrome's tab search and Firefox's add-ons manager, so E, the archive
  // key every mail client uses, on the ⌥ run the rest of these chat chords sit on.
  def("archiveChat", "Mod+Alt+KeyE"),
  def("newStandaloneChat", "Mod+Alt+KeyO"),
  // ⌃⇧U off macOS is GTK's hex entry, hard-coded in GtkIMContextSimple and
  // bound again by IBus, so a focused composer is where it would be fought
  // over. U stays the mnemonic on the ⌥ run, beside Clear all unreads.
  def("markChatUnread", "Mod+Shift+KeyU", {
    nonMacDefaultBinding: "Mod+Alt+KeyU",
  }),
  // ⌥⌘P is Chrome's Page Setup on macOS, so P moves to the ⌃⇧ run there, the
  // same swap the API monitor and the composer pair make.
  def("togglePinChat", "Ctrl+Shift+KeyP", {
    nonMacDefaultBinding: "Mod+Alt+KeyP",
  }),
  // Archive, pin and mark unread above already act on the selection when
  // there is one, so these only cover what a selection alone needs.
  def("selectAllChats", "Mod+Alt+KeyS"),
  // Escape clears the selection too, from the sidebar's own listener: this
  // registry cannot bind it, since declining a tool call owns bare Escape.
  def("clearChatSelection", null),
  // Unassigned on purpose: nothing that deletes chats should ship on a chord.
  def("deleteSelectedChats", null),
  def("nextRecentlyViewedChat", "Ctrl+Tab", {
    nonMacDefaultBinding: "Mod+Tab",
  }),
  // No arrow alternate: ⌥⌘→ is Chrome's next tab, and off macOS the same chord
  // reads as Ctrl+Alt+→, desktop switching on GNOME and KDE and screen rotation
  // on Intel graphics. Taken everywhere, so the bracket pair carries these.
  def("nextChat", "Mod+Shift+BracketRight"),
  def("nextChatNeedingAttention", "Mod+Alt+KeyA"),
  def("previousRecentlyViewedChat", "Ctrl+Shift+Tab", {
    nonMacDefaultBinding: "Mod+Shift+Tab",
  }),
  def("previousChat", "Mod+Shift+BracketLeft"),
  ...RECENT_SLOT_DEFS,

  // -- Workspaces ---------------------------------------------------------
  ...WORKSPACE_DEFS,

  // -- Panels and app-level actions ---------------------------------------
  // ⌥⌘U is view source on macOS, so U keeps its mnemonic on ⌃⇧ there instead,
  // the same swap the composer pair below makes. Off macOS it gives U up
  // altogether: three actions wanted that letter and only two chords carry it
  // safely there, so the two that mean "unread" have them and this one, whose
  // U was never a mnemonic, takes M for monitor.
  def("toggleApiMonitor", "Ctrl+Shift+KeyU", {
    nonMacDefaultBinding: "Mod+Alt+Shift+KeyM",
  }),
  def("toggleSidebar", "Mod+KeyB"),
  def("openMcpServers", null),
  // ⇧Esc is Chrome's and Edge's task manager off macOS.
  def("clearAllUnreads", "Shift+Escape", {
    nonMacDefaultBinding: "Mod+Alt+Shift+KeyU",
  }),
  // Desktop signs out through the OS account menu, and the sidebar hides its
  // own logout item there, so this row would bind a chord that cannot fire.
  def("logOut", null, { webOnly: true }),
  def("openSettings", "Mod+Comma"),
  // Bare ⏎ / Esc: both register only while a tool call is waiting.
  def("approveToolRequest", "Enter", { allowBareKey: true }),
  def("declineToolRequest", "Escape", { allowBareKey: true }),

  // -- Composer -----------------------------------------------------------
  def("attachFiles", null),
  def("cycleReasoningEffort", null),
  def("decreaseReasoningEffort", null),
  def("increaseReasoningEffort", null),
  // Off macOS the ⇧ pair collides with Chrome's profile switcher and its
  // bookmark-every-tab, so both move to Alt.
  def("openModelPicker", "Ctrl+Shift+KeyM", {
    nonMacDefaultBinding: "Mod+Alt+KeyM",
  }),
  def("openProjectPicker", "Mod+Alt+Shift+KeyO"),
  def("startDictation", "Ctrl+Shift+KeyD", {
    nonMacDefaultBinding: "Mod+Alt+KeyV",
  }),
  def("sendMessage", null),
  def("toggleFastMode", null),

  // -- Chat actions -------------------------------------------------------
  def("copyChatAsMarkdown", null),
  // ⌥⌘C is Firefox's element picker and Safari's console, so C moves the same
  // way U did above.
  def("copySessionId", "Ctrl+Shift+KeyC", {
    nonMacDefaultBinding: "Mod+Alt+KeyC",
  }),
  def("forkChat", null),
  // No ⇧⌘P alternate: it is the command-menu chord everywhere else, but in
  // Firefox it opens a private window, and ⌘K is the one people reach for.
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

/** The chord this build ships for `id`'s `slot` on this platform. */
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
 * Chords the browser owns. Tab and window management never reaches the page at
 * all; the rest is cancellable, but taking it breaks the browser's own action.
 * Both work in the desktop build, so they still ship as defaults and the tab
 * flags them on the web.
 *
 * What gets in, here and in the two platform sets below: a chord the browser
 * takes from every user out of the box. These sets drive a warning the user
 * reads and a test no default may fail, and both are only worth anything if
 * the chord really is gone on the machine in front of them. A chord that only
 * binds once a hidden developer menu is switched on does not qualify -- Safari
 * puts Empty Caches on ⌥⌘E and reloads from origin on ⌥⌘R, but only for
 * someone who went and enabled the Develop menu, so warning every macOS user
 * about them would be telling almost all of them something untrue. Where a
 * default and an opt-in chord do collide, the tab that raised the menu is also
 * the tab that rebinds.
 */
const BROWSER_RESERVED_VALUES = new Set<string>([
  "Mod+KeyN",
  "Mod+Shift+KeyN",
  "Mod+KeyT",
  "Mod+Shift+KeyT",
  "Mod+KeyW",
  "Mod+Shift+KeyW",
  "Mod+KeyL",
  "Mod+KeyR",
  "Mod+Shift+KeyR",
  "Mod+KeyP",
  // Firefox's new private window, on both platforms. Chrome puts that on
  // ⇧⌘N, which is already here; this is the other half of the same pair.
  "Mod+Shift+KeyP",
  // Chrome's tab search, and Firefox's add-ons manager.
  "Mod+Shift+KeyA",
  "Mod+Tab",
  "Mod+Shift+Tab",
  "Ctrl+Tab",
  "Ctrl+Shift+Tab",
  ...Array.from({ length: 9 }, (_, i) => `Mod+Digit${i + 1}`),
]);

/**
 * Owned on macOS only, where ⌥⌘ is the browsers' own run. From Chrome's
 * shortcut list: U view source, I dev tools, J console, B bookmarks, N split
 * view, F search, and the arrows for tabs and toolbar focus. Firefox adds C,
 * its element picker, and K, its console. Off macOS the same values read as
 * Ctrl+Alt, which none of them claim.
 */
const MAC_RESERVED_VALUES = new Set<string>([
  "Mod+Alt+KeyU",
  "Mod+Alt+KeyI",
  "Mod+Alt+KeyJ",
  "Mod+Alt+KeyB",
  "Mod+Alt+KeyN",
  "Mod+Alt+KeyF",
  "Mod+Alt+KeyC",
  "Mod+Alt+KeyK",
  // Page Setup, which Chrome lists beside ⌘P on its own shortcuts page.
  "Mod+Alt+KeyP",
  // Safari's Show Next Tab and Show Previous Tab, per Apple's own shortcut
  // list, and Chrome carries the same pair on macOS. Off macOS they read as
  // Ctrl+Shift+bracket, which no browser claims. The chat walk ships on them
  // for the desktop build, where nothing intercepts them; reserving them is
  // what puts the warning in front of a web user who is about to rebind.
  "Mod+Shift+BracketLeft",
  "Mod+Shift+BracketRight",
  "Mod+Alt+ArrowLeft",
  "Mod+Alt+ArrowRight",
  "Mod+Alt+ArrowUp",
  "Mod+Alt+ArrowDown",
]);

/** Owned off macOS only: Chrome's task manager, and Firefox on Alt. */
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

/**
 * Keys the browser turns into a click on whatever control has focus. A chord
 * built from one of these with no modifier has to leave that click alone, or
 * pressing Enter on a focused Deny button would run the shortcut instead and
 * preventDefault would cancel the button's own activation.
 */
const ACTIVATION_CODES = new Set(["Enter", "NumpadEnter", "Space"]);

const ACTIVATABLE_TAGS = new Set(["BUTTON", "A", "SUMMARY", "SELECT", "OPTION"]);

/** True when this chord is a bare activation key and focus is on a control. */
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
    getModifierState?: (key: string) => boolean;
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
  // matchesBinding will not fire an AltGr chord, so do not record one.
  if (!mac && isAltGraphEvent(event)) return null;
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

/**
 * AltGr rather than a real Ctrl+Alt. AltGr reports both, so typing ą or € would
 * otherwise fire every Ctrl+Alt chord and eat the character. Callers gate this
 * on being off macOS: Option reports AltGraph too, and it is a plain Alt.
 */
function isAltGraphEvent(event: {
  getModifierState?: (key: string) => boolean;
}): boolean {
  return event.getModifierState?.("AltGraph") === true;
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
    getModifierState?: (key: string) => boolean;
  },
  mac = isMacPlatform(),
): boolean {
  const code = event.code || keyToCode(event.key ?? "");
  if (code !== binding.code) return false;
  // Off macOS a Ctrl chord is unreachable, and without this a value stored on
  // a Mac would fall through the checks below and fire on the bare key.
  if (!mac && binding.ctrl) return false;
  if (!mac && binding.alt && isAltGraphEvent(event)) return false;
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
 * recorder refuses it. Function keys are self-contained, and an action gated on
 * an on-screen prompt (`allowBareKey`) may take any key.
 */
export function isAcceptableBinding(
  binding: ShortcutBinding,
  allowBareKey = false,
): boolean {
  if (binding.mod || binding.ctrl || binding.alt) return true;
  // Tab moves focus, and a chord consumes the key it answers to. Bound bare,
  // even on a prompt-gated action, it makes that prompt's own buttons
  // unreachable by keyboard. Held with a modifier it is a chord like any
  // other, which is where the recently-viewed walk sits.
  if (binding.code === "Tab") return false;
  if (allowBareKey) return true;
  if (/^F\d{1,2}$/.test(binding.code)) return true;
  // Escape is self-contained too, but bare it belongs to declining a tool call
  // and is the way out of the recorder. Held with Shift it is free, which is
  // where clearAllUnreads sits. Anything else has to say so with allowBareKey.
  return binding.code === "Escape" && binding.shift;
}
