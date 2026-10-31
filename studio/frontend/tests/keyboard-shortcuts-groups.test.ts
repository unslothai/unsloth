// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  SHORTCUT_DEFS,
  SHORTCUT_GROUPS,
} from "../src/features/settings/lib/keyboard-shortcuts.ts";
import {
  assignedBinding,
  findConflicts,
  isShortcutEnabled,
  migrateStoredOverrides,
  resolveBinding,
  useKeyboardShortcutsStore,
} from "../src/features/settings/stores/keyboard-shortcuts-store.ts";
import { en } from "../src/i18n/locales/en.ts";

import { readSrc } from "./helpers/kit.ts";

test("every action sits in exactly one section of the shortcuts tab", () => {
  const seen = new Map<string, string>();
  for (const group of SHORTCUT_GROUPS) {
    for (const id of group.ids) {
      assert.equal(
        seen.has(id),
        false,
        `${id} is in ${seen.get(id)} and ${group.id}`,
      );
      seen.set(id, group.id);
    }
    assert.ok(
      (en.settings.keyboardShortcuts.groups as Record<string, string>)[
        group.id
      ],
      `${group.id} has no English heading`,
    );
  }
  for (const def of SHORTCUT_DEFS) {
    assert.ok(
      seen.has(def.id),
      `${def.id} is in no section, so the tab never shows it`,
    );
  }
  assert.equal(seen.size, SHORTCUT_DEFS.length);
});

test("a switched-off action keeps its chord but stops answering to it", () => {
  const store = useKeyboardShortcutsStore.getState();
  store.resetAll();
  const before = resolveBinding({}, "searchChats");
  assert.ok(before);

  store.setEnabled("searchChats", false);
  let { overrides } = useKeyboardShortcutsStore.getState();
  assert.equal(isShortcutEnabled(overrides, "searchChats"), false);
  assert.equal(resolveBinding(overrides, "searchChats"), null);
  assert.equal(assignedBinding(overrides, "searchChats"), before);

  store.setEnabled("searchChats", true);
  ({ overrides } = useKeyboardShortcutsStore.getState());
  assert.deepEqual(
    overrides,
    {},
    "switching back on leaves nothing overridden",
  );
});

test("recording a chord switches the action back on", () => {
  const store = useKeyboardShortcutsStore.getState();
  store.resetAll();
  store.setEnabled("renameChat", false);
  store.setBinding("renameChat", "primary", "Mod+Alt+KeyN");
  const { overrides } = useKeyboardShortcutsStore.getState();
  assert.deepEqual(overrides.renameChat, { primary: "Mod+Alt+KeyN" });
  assert.equal(resolveBinding(overrides, "renameChat"), "Mod+Alt+KeyN");
  store.resetAll();
});

test("a switched-off action no longer clashes with the one sharing its chord", () => {
  const overrides = migrateStoredOverrides({
    renameChat: { primary: "Mod+KeyK" },
  });
  assert.ok(findConflicts(overrides).has("renameChat"));
  const off = migrateStoredOverrides({
    renameChat: { primary: "Mod+KeyK", disabled: true },
  });
  assert.deepEqual(off.renameChat, { primary: "Mod+KeyK", disabled: true });
  assert.equal(findConflicts(off).has("renameChat"), false);
  assert.deepEqual(migrateStoredOverrides({ renameChat: { disabled: true } }), {
    renameChat: { disabled: true },
  });
});

test("the recorder clears a slot on a bare Backspace or Delete", () => {
  const tab = readSrc("features/settings/tabs/keyboard-shortcuts-tab.tsx");
  assert.match(
    tab,
    /if \(bare && \(binding\.code === "Backspace" \|\| binding\.code === "Delete"\)\) \{\s*clearBinding\(recording\.id, recording\.slot\);/,
  );
  // Leaving the box any other way ends recording, so no stale listener keeps swallowing keys.
  assert.match(tab, /onBlur=\{stopRecording\}/);
});

test("both composers open their + menu on the chord, never from behind a dialog", () => {
  for (const path of ["components/assistant-ui/thread.tsx", "features/chat/shared-composer.tsx"]) {
    const src = readSrc(path);
    const at = src.indexOf('"openComposerMenu",');
    assert.notEqual(at, -1, `${path} does not register openComposerMenu`);
    assert.match(
      src.slice(at, at + 400),
      /if \(plusOpen\) return setPlusMenuOpen\(false\);\s*if \(!isSurfaceInForeground\(COMPOSER_INPUT_SELECTOR\)\) return;\s*setPlusMenuOpen\(true\);/,
    );
    assert.match(src, /<DropdownMenu open=\{plusOpen\} onOpenChange=\{setPlusMenuOpen\}>/);
  }
});
