// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_CUSTOMIZATION,
  sanitizeCustomization,
  useAppearanceCustomStore,
} from "../src/features/settings/stores/appearance-custom-store.ts";
import { createSettingsSearchIndex, SETTINGS_SEARCH_KEYWORDS } from "../src/features/settings/settings-search.ts";

test("legacy and malformed preferences preserve the enabled mascot default", () => {
  assert.equal(DEFAULT_CUSTOMIZATION.showMascots, true);
  for (const showMascots of [undefined, null, "false", 0, true]) {
    assert.equal(sanitizeCustomization({ showMascots }).showMascots, true);
  }
  assert.equal(sanitizeCustomization({ showMascots: false }).showMascots, false);
});

test("mascot changes persist, hydrate with older payloads, and reset", async () => {
  const entries = new Map<string, string>();
  const storage = {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => { entries.set(key, value); },
    removeItem: (key: string) => { entries.delete(key); },
  };
  Object.assign(globalThis, { window: { localStorage: storage } });
  useAppearanceCustomStore.getState().patch({ showMascots: false });
  const key = "unsloth_appearance_customization";
  assert.equal(JSON.parse(entries.get(key)!).state.customization.showMascots, false);
  useAppearanceCustomStore.setState({ customization: DEFAULT_CUSTOMIZATION });
  storage.setItem(key, JSON.stringify({ version: 8, state: { customization: { showMascots: false } } }));
  await useAppearanceCustomStore.persist.rehydrate();
  assert.equal(useAppearanceCustomStore.getState().customization.showMascots, false);
  storage.setItem(key, JSON.stringify({ version: 8, state: { customization: { pointerCursors: true } } }));
  await useAppearanceCustomStore.persist.rehydrate();
  assert.equal(useAppearanceCustomStore.getState().customization.showMascots, true);
  assert.equal(useAppearanceCustomStore.getState().customization.pointerCursors, true);
  useAppearanceCustomStore.getState().patch({ showMascots: false });
  useAppearanceCustomStore.getState().resetAll();
  assert.equal(useAppearanceCustomStore.getState().customization.showMascots, true);
});

test("mascot and sloth searches reach Appearance on desktop and web", () => {
  const key = "settings.appearance.custom.mascots.label";
  for (const desktop of [true, false]) {
    assert.ok(createSettingsSearchIndex({ desktop, closeToTray: desktop }).appearance.includes(key));
  }
  assert.equal(SETTINGS_SEARCH_KEYWORDS[key], "settings.appearance.custom.mascots.keywords");
});
