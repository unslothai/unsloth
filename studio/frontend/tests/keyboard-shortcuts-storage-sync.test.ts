// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const stored = new Map<string, string>();
const listeners: Array<(event: { key: string | null }) => void> = [];
(globalThis as { window?: unknown }).window = {
  localStorage: {
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => stored.set(key, value),
    removeItem: (key: string) => stored.delete(key),
  },
  addEventListener: (
    type: string,
    listener: (event: { key: string | null }) => void,
  ) => {
    if (type === "storage") listeners.push(listener);
  },
};

const { KEYBOARD_SHORTCUTS_STORAGE_KEY, useKeyboardShortcutsStore } =
  await import("../src/features/settings/stores/keyboard-shortcuts-store.ts");

const fire = (key: string | null) => {
  for (const listener of listeners) listener({ key });
};

test("an edit in another tab reaches this one before it writes again", () => {
  assert.equal(listeners.length, 1);
  useKeyboardShortcutsStore
    .getState()
    .setBinding("searchChats", "primary", "Mod+KeyJ");
  stored.set(
    KEYBOARD_SHORTCUTS_STORAGE_KEY,
    JSON.stringify({ openSettings: { primary: "Mod+Alt+Comma" } }),
  );
  fire(KEYBOARD_SHORTCUTS_STORAGE_KEY);
  assert.deepEqual(useKeyboardShortcutsStore.getState().overrides, {
    openSettings: { primary: "Mod+Alt+Comma" },
  });
  useKeyboardShortcutsStore
    .getState()
    .setBinding("renameChat", "primary", "Mod+Alt+KeyT");
  assert.deepEqual(
    JSON.parse(stored.get(KEYBOARD_SHORTCUTS_STORAGE_KEY) ?? "{}"),
    {
      openSettings: { primary: "Mod+Alt+Comma" },
      renameChat: { primary: "Mod+Alt+KeyT" },
    },
  );
});

test("other keys are ignored, and a cleared storage resets", () => {
  const before = useKeyboardShortcutsStore.getState().overrides;
  stored.set(KEYBOARD_SHORTCUTS_STORAGE_KEY, "{}");
  fire("unsloth_theme");
  assert.equal(useKeyboardShortcutsStore.getState().overrides, before);
  stored.clear();
  fire(null);
  assert.deepEqual(useKeyboardShortcutsStore.getState().overrides, {});
});
