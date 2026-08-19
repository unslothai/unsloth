// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  SHORTCUT_DEFS,
  bindingFromEvent,
  formatBindingLabel,
  formatBindingValue,
  isAcceptableBinding,
  isShortcutId,
  matchesBinding,
  parseBinding,
} from "../src/features/settings/lib/keyboard-shortcuts.ts";
import {
  KEYBOARD_SHORTCUTS_STORAGE_KEY,
  findConflicts,
  resolveAllBindings,
  resolveBinding,
  shortcutOwningBinding,
} from "../src/features/settings/stores/keyboard-shortcuts-store.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";

function keyEvent(
  code: string,
  mods: Partial<{
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
  }> = {},
) {
  return {
    code,
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    ...mods,
  };
}

test("a binding round-trips through serialize and parse", () => {
  const binding = {
    code: "KeyO",
    mod: true,
    ctrl: false,
    shift: true,
    alt: false,
  };
  const value = formatBindingValue(binding);
  assert.equal(value, "Mod+Shift+KeyO");
  assert.deepEqual(parseBinding(value), binding);
});

test("parse rejects junk, empties and modifier-only values", () => {
  assert.equal(parseBinding(null), null);
  assert.equal(parseBinding(""), null);
  assert.equal(parseBinding("Mod+ShiftLeft"), null);
  assert.equal(parseBinding("Hyper+KeyO"), null);
});

test("every shipped default parses", () => {
  for (const def of SHORTCUT_DEFS) {
    if (def.defaultBinding === null) continue;
    assert.ok(
      parseBinding(def.defaultBinding),
      `${def.id} has an unparseable default: ${def.defaultBinding}`,
    );
  }
});

test("matching is exact about modifiers", () => {
  const binding = parseBinding("Mod+Shift+KeyO");
  assert.ok(binding);
  // Non-mac, so Mod is Ctrl.
  assert.ok(
    matchesBinding(
      binding,
      keyEvent("KeyO", { ctrlKey: true, shiftKey: true }),
      false,
    ),
  );
  // Missing Shift, extra Alt, and wrong key must all miss.
  assert.equal(
    matchesBinding(binding, keyEvent("KeyO", { ctrlKey: true }), false),
    false,
  );
  assert.equal(
    matchesBinding(
      binding,
      keyEvent("KeyO", { ctrlKey: true, shiftKey: true, altKey: true }),
      false,
    ),
    false,
  );
  assert.equal(
    matchesBinding(
      binding,
      keyEvent("KeyP", { ctrlKey: true, shiftKey: true }),
      false,
    ),
    false,
  );
});

test("off-platform Meta does not satisfy a Mod binding", () => {
  const binding = parseBinding("Mod+KeyB");
  assert.ok(binding);
  // Windows key held instead of Ctrl, on a non-mac platform.
  assert.equal(
    matchesBinding(binding, keyEvent("KeyB", { metaKey: true }), false),
    false,
  );
});

test("on macOS Mod is Cmd, and a bare Ctrl is not a substitute", () => {
  const binding = parseBinding("Mod+KeyB");
  assert.ok(binding);
  assert.ok(matchesBinding(binding, keyEvent("KeyB", { metaKey: true }), true));
  assert.equal(
    matchesBinding(binding, keyEvent("KeyB", { ctrlKey: true }), true),
    false,
  );
  // Ctrl is bindable in its own right on macOS.
  const ctrlBinding = parseBinding("Ctrl+KeyB");
  assert.ok(ctrlBinding);
  assert.ok(
    matchesBinding(ctrlBinding, keyEvent("KeyB", { ctrlKey: true }), true),
  );
});

test("recording a chord ignores a lone modifier", () => {
  assert.equal(
    bindingFromEvent(keyEvent("ShiftLeft", { shiftKey: true }), false),
    null,
  );
  const binding = bindingFromEvent(keyEvent("KeyK", { ctrlKey: true }), false);
  assert.ok(binding);
  assert.equal(formatBindingValue(binding), "Mod+KeyK");
  // The same physical chord on macOS is Cmd, not Ctrl.
  const macBinding = bindingFromEvent(
    keyEvent("KeyK", { metaKey: true }),
    true,
  );
  assert.ok(macBinding);
  assert.equal(formatBindingValue(macBinding), "Mod+KeyK");
});

test("a bare letter is refused but function keys stand alone", () => {
  assert.equal(
    isAcceptableBinding({
      code: "KeyK",
      mod: false,
      ctrl: false,
      shift: false,
      alt: false,
    }),
    false,
  );
  assert.ok(
    isAcceptableBinding({
      code: "F5",
      mod: false,
      ctrl: false,
      shift: false,
      alt: false,
    }),
  );
});

test("labels use each platform's own modifier notation", () => {
  const binding = parseBinding("Mod+Shift+KeyO");
  assert.ok(binding);
  assert.equal(formatBindingLabel(binding, true), "⇧⌘O");
  assert.equal(formatBindingLabel(binding, false), "Ctrl+Shift+O");
  const comma = parseBinding("Mod+Comma");
  assert.ok(comma);
  assert.equal(formatBindingLabel(comma, true), "⌘,");
});

test("an override wins, and null means unassigned", () => {
  assert.equal(resolveBinding({}, "toggleSidebar"), "Mod+KeyB");
  assert.equal(
    resolveBinding({ toggleSidebar: "Mod+Alt+KeyB" }, "toggleSidebar"),
    "Mod+Alt+KeyB",
  );
  // Present-but-null is a deliberate clear, not a fallback to the default.
  assert.equal(resolveBinding({ toggleSidebar: null }, "toggleSidebar"), null);
});

test("defaults ship without conflicts", () => {
  assert.equal(findConflicts({}).size, 0);
  const all = resolveAllBindings({});
  assert.equal(all.newChat, "Mod+Shift+KeyO");
});

test("two actions on one chord are both flagged", () => {
  const conflicts = findConflicts({ toggleSidebar: "Mod+KeyK" });
  assert.deepEqual(
    [...conflicts].sort(),
    ["searchChats", "toggleSidebar"].sort(),
  );
});

test("cleared actions never count as conflicting", () => {
  const conflicts = findConflicts({ toggleSidebar: null, searchChats: null });
  assert.equal(conflicts.size, 0);
});

test("ids from an older build are rejected", () => {
  assert.ok(isShortcutId("newChat"));
  assert.equal(isShortcutId("someRemovedAction"), false);
});

test("a contested chord is owned by the earlier action in registry order", () => {
  // searchChats is declared before toggleSidebar, so it keeps Mod+K.
  const overrides = { toggleSidebar: "Mod+KeyK" };
  assert.equal(shortcutOwningBinding(overrides, "Mod+KeyK"), "searchChats");
  // The loser keeps its own binding elsewhere, so ownership is per chord.
  assert.equal(shortcutOwningBinding(overrides, "Mod+Comma"), "openSettings");
});

test("exactly one owner exists per contested chord", () => {
  const overrides = { toggleSidebar: "Mod+KeyK", newChat: "Mod+KeyK" };
  const contested = [...findConflicts(overrides)];
  assert.equal(contested.length, 3);
  const owners = new Set(
    contested.map(() => shortcutOwningBinding(overrides, "Mod+KeyK")),
  );
  assert.equal(owners.size, 1);
  // newChat is declared first, so it wins over searchChats and toggleSidebar.
  assert.equal(shortcutOwningBinding(overrides, "Mod+KeyK"), "newChat");
});

test("an unbound or unclaimed chord has no owner", () => {
  assert.equal(shortcutOwningBinding({}, null), null);
  assert.equal(shortcutOwningBinding({}, "Mod+Alt+KeyZ"), null);
  // A cleared action does not own the chord it used to have.
  assert.equal(shortcutOwningBinding({ searchChats: null }, "Mod+KeyK"), null);
});

// Reset-all is the documented escape hatch. A chord bound to something the
// browser eats leaves the Shortcuts tab itself hard to reach, so the General
// reset has to cover this key or the user is stuck with it.
test("Reset all local preferences clears the rebound chords", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
    "utf8",
  );
  const keys = source.slice(
    source.indexOf("const PREFS_KEYS"),
    source.indexOf("];", source.indexOf("const PREFS_KEYS")),
  );
  assert.ok(keys, "PREFS_KEYS moved; this contract needs updating");
  assert.ok(
    keys.includes("KEYBOARD_SHORTCUTS_STORAGE_KEY") ||
      keys.includes(`"${KEYBOARD_SHORTCUTS_STORAGE_KEY}"`),
    `${KEYBOARD_SHORTCUTS_STORAGE_KEY} missing from PREFS_KEYS`,
  );
});

// Every overlay carries the tab label and every row, or a non-English install
// shows an English word in the middle of a translated settings dialog.
test("every locale overlay carries the shortcut strings", async () => {
  const locales = [
    "ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pt-br", "ru", "zh-CN",
  ];
  for (const locale of locales) {
    const source = await readFile(
      new URL(`../src/i18n/locales/${locale}.ts`, import.meta.url),
      "utf8",
    );
    const at = source.indexOf("keyboardShortcuts: {");
    assert.notEqual(
      at,
      -1,
      `${locale} is missing the settings.keyboardShortcuts subtree`,
    );
    const subtree = source.slice(at, source.indexOf("\n    },", at));
    for (const key of ["title", "resetAll", "conflictShadowed", "groups"]) {
      assert.ok(
        subtree.includes(`${key}:`),
        `${locale} is missing settings.keyboardShortcuts.${key}`,
      );
    }
    for (const def of SHORTCUT_DEFS) {
      assert.ok(
        subtree.includes(`${def.id}: {`),
        `${locale} is missing settings.keyboardShortcuts.actions.${def.id}`,
      );
    }
  }
});

test("every settings tab survives a reload", () => {
  // The persisted-tab check reads this same list, so a tab added to the union
  // alone can no longer be rejected back to General.
  assert.ok(SETTINGS_TABS.includes("keyboard-shortcuts"));
  assert.equal(new Set(SETTINGS_TABS).size, SETTINGS_TABS.length);
});

test("a Super chord off macOS records nothing rather than a different chord", () => {
  // matchesBinding rejects a non-mac event carrying Meta, so there is nowhere to
  // put it. Persisting Alt+K for a user who pressed Super+Alt+K would assign an
  // action to a chord they did not choose, and fire it on Alt+K alone.
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true, altKey: true }), false),
    null,
  );
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true }), false),
    null,
  );
  // Ctrl+Meta is likewise not a Mod chord with the Meta quietly dropped.
  assert.equal(
    bindingFromEvent(keyEvent("KeyK", { metaKey: true, ctrlKey: true }), false),
    null,
  );
  // The same chords on macOS are ordinary Cmd bindings and still record.
  const mac = bindingFromEvent(
    keyEvent("KeyK", { metaKey: true, altKey: true }),
    true,
  );
  assert.ok(mac);
  assert.equal(formatBindingValue(mac), "Mod+Alt+KeyK");
});

// The sidebar search tooltip and the Settings menu row are the two hints a
// user sees outside the shortcuts tab. Hard-coded, they keep advertising the
// shipped chord after a rebind, and a dead one after a clear.
test("the sidebar hints render the bound chord, not the shipped default", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  for (const literal of ['"⌘K"', '"Ctrl+K"', "<DropdownMenuShortcut>⌘,"]) {
    assert.ok(
      !source.includes(literal),
      `app-sidebar still hard-codes ${literal}`,
    );
  }
  assert.ok(source.includes('useShortcutLabel("searchChats")'));
  assert.ok(source.includes('useShortcutLabel("openSettings")'));
  // Unassigned actions must drop the hint rather than render an empty key cap.
  assert.ok(source.includes("{searchShortcutLabel && ("));
  assert.ok(source.includes("{settingsShortcutLabel && ("));
});

test("a hint label follows the override and disappears when cleared", () => {
  const label = (overrides: Parameters<typeof resolveBinding>[0], id: "searchChats" | "openSettings") => {
    const binding = parseBinding(resolveBinding(overrides, id));
    return binding ? formatBindingLabel(binding, false) : null;
  };
  assert.equal(label({}, "searchChats"), "Ctrl+K");
  assert.equal(label({ searchChats: "Mod+Shift+KeyF" }, "searchChats"), "Ctrl+Shift+F");
  assert.equal(label({ searchChats: null }, "searchChats"), null);
  assert.equal(label({}, "openSettings"), "Ctrl+,");
});
