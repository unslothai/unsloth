// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  formatBindingValue,
  parseBinding,
  SHORTCUT_SLOTS,
  type ShortcutId,
} from "@/features/settings/lib/keyboard-shortcuts";
import {
  resolveBinding,
  shortcutOwningBinding,
} from "@/features/settings/stores/keyboard-shortcuts-store";
import type { HelpAction } from "@/components/help-actions";
import type { SettingsTab } from "@/features/settings";

/** Go > Settings opens each page of the Settings dialog. */
export type SettingsMenuAction = `settings-${SettingsTab}`;

/** Actions the desktop File, View, Go and Help menus send (src-tauri/src/app_menu.rs). */
export type AppMenuAction =
  | HelpAction
  | SettingsMenuAction
  | "new-chat"
  | "new-temporary-chat"
  | "open-folder"
  | "toggle-sidebar"
  | "find"
  | "previous-chat"
  | "next-chat"
  | "back"
  | "forward"
  | "zoom-in"
  | "zoom-out"
  | "actual-size"
  | "go-chat"
  | "go-projects"
  | "go-library"
  | "go-hub"
  | "go-train"
  | "go-recipes"
  | "go-images"
  | "go-video"
  | "go-audio"
  | "go-export";

/** The chord each item shows (as in src-tauri/src/app_menu.rs). An item for a web shortcut follows
 *  that shortcut's binding; the others keep theirs unless a web shortcut has taken it. An item
 *  left out, as the Settings pages are, has none. */
export const MENU_CHORDS: Partial<
  Record<AppMenuAction, { shortcut?: ShortcutId; chord?: string }>
> = {
  "new-chat": { shortcut: "newChat", chord: "Mod+KeyN" },
  "new-temporary-chat": { shortcut: "newTemporaryChat", chord: "Mod+Shift+KeyN" },
  "open-folder": { chord: "Mod+KeyO" },
  "toggle-sidebar": { shortcut: "toggleSidebar", chord: "Mod+KeyB" },
  "find": { shortcut: "findInPage", chord: "Mod+KeyF" },
  "previous-chat": { shortcut: "previousChat", chord: "Mod+Shift+BracketLeft" },
  "next-chat": { shortcut: "nextChat", chord: "Mod+Shift+BracketRight" },
  "back": { chord: "Mod+BracketLeft" },
  "forward": { chord: "Mod+BracketRight" },
  "zoom-in": { chord: "Mod+Equal" },
  "zoom-out": { chord: "Mod+Minus" },
  "actual-size": { chord: "Mod+Digit0" },
  "go-chat": { shortcut: "switchToChat", chord: "Ctrl+Digit1" },
  "go-projects": { shortcut: "switchToProjects", chord: "Ctrl+Digit2" },
  "go-library": {},
  "go-hub": { shortcut: "switchToHub", chord: "Ctrl+Digit3" },
  "go-train": { shortcut: "switchToTrain", chord: "Ctrl+Digit4" },
  "go-recipes": { shortcut: "switchToRecipes", chord: "Ctrl+Digit5" },
  "go-images": { shortcut: "switchToImages", chord: "Ctrl+Digit6" },
  "go-video": { shortcut: "switchToVideo", chord: "Ctrl+Digit7" },
  "go-audio": { shortcut: "switchToAudio", chord: "Ctrl+Digit8" },
  "go-export": { shortcut: "switchToExport", chord: "Ctrl+Digit9" },
  "help-documentation": {},
  "help-keyboard-shortcuts": { shortcut: "openKeyboardShortcuts", chord: "Mod+Slash" },
  "help-whats-new": {},
  "help-troubleshooting": {},
  "help-system-status": {},
  "help-send-feedback": {},
};

/** Chords the native macOS menu keeps (Quit, Close, Minimize, Hide, Hide Others, Edit, Enter
 *  Full Screen). Two items on one key equivalent means one silently loses, so ours skip them. */
export const NATIVE_MENU_CHORDS: ReadonlySet<string> = new Set([
  "Mod+KeyQ",
  "Mod+KeyW",
  "Mod+KeyM",
  "Mod+KeyH",
  "Mod+Alt+KeyH",
  "Mod+KeyZ",
  "Mod+Shift+KeyZ",
  "Mod+KeyX",
  "Mod+KeyC",
  "Mod+KeyV",
  "Mod+KeyA",
  "Mod+Ctrl+KeyF",
]);

/** A binding as a native accelerator. Only Cmd or Ctrl chords, since a bare key there would take
 *  typing away from text fields, and none a native item already has. */
function toAccelerator(value: string | null): string | null {
  const binding = parseBinding(value);
  if (!binding || !(binding.mod || binding.ctrl)) return null;
  if (NATIVE_MENU_CHORDS.has(formatBindingValue(binding))) return null;
  return [
    binding.mod && "CmdOrCtrl",
    binding.ctrl && "Ctrl",
    binding.alt && "Alt",
    binding.shift && "Shift",
    binding.code,
  ]
    .filter(Boolean)
    .join("+");
}

/** What each menu item should show for these shortcut overrides. */
export function menuAccelerators(
  overrides: Parameters<typeof resolveBinding>[0],
): Partial<Record<AppMenuAction, string | null>> {
  const out: Partial<Record<AppMenuAction, string | null>> = {};
  for (const [action, { shortcut, chord }] of Object.entries(MENU_CHORDS) as [
    AppMenuAction,
    { shortcut?: ShortcutId; chord?: string },
  ][]) {
    if (!shortcut) {
      out[action] =
        !chord || shortcutOwningBinding(overrides, chord) ? null : toAccelerator(chord);
      continue;
    }
    const owned = SHORTCUT_SLOTS.map((slot) => resolveBinding(overrides, shortcut, slot)).filter(
      (value): value is string =>
        Boolean(value) && shortcutOwningBinding(overrides, value) === shortcut,
    );
    const shown = chord && owned.includes(chord) ? chord : owned.find((value) => toAccelerator(value));
    out[action] = toAccelerator(shown ?? null);
  }
  return out;
}
