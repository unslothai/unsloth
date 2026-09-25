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

/** Actions the desktop File and View menus send (src-tauri/src/app_menu.rs). */
export type AppMenuAction =
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
  | "actual-size";

/** The chord each item shows (as in src-tauri/src/app_menu.rs). An item for a web shortcut follows
 *  that shortcut's binding; the others keep theirs unless a web shortcut has taken it. */
export const MENU_CHORDS: Record<AppMenuAction, { shortcut?: ShortcutId; chord: string }> = {
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
): Record<AppMenuAction, string | null> {
  const out = {} as Record<AppMenuAction, string | null>;
  for (const action of Object.keys(MENU_CHORDS) as AppMenuAction[]) {
    const { shortcut, chord } = MENU_CHORDS[action];
    if (!shortcut) {
      out[action] = shortcutOwningBinding(overrides, chord) ? null : toAccelerator(chord);
      continue;
    }
    const owned = SHORTCUT_SLOTS.map((slot) => resolveBinding(overrides, shortcut, slot)).filter(
      (value): value is string =>
        Boolean(value) && shortcutOwningBinding(overrides, value) === shortcut,
    );
    const shown = owned.includes(chord) ? chord : owned.find((value) => toAccelerator(value));
    out[action] = toAccelerator(shown ?? null);
  }
  return out;
}
