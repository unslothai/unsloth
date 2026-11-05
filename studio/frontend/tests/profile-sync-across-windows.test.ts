// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

const KEY = "unsloth_user_profile";
const APPEARANCE_KEY = "unsloth_appearance_customization";
const data = new Map<string, string>();
type StorageEvent = { key: string | null; newValue: string | null };
const listeners: ((event: StorageEvent) => void)[] = [];
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: {
    localStorage,
    addEventListener: (
      type: string,
      listener: (event: StorageEvent) => void,
    ) => {
      if (type === "storage") listeners.push(listener);
    },
  },
});

const { useUserProfileStore } = await import(
  "../src/features/profile/stores/user-profile-store.ts"
);
const { useAppearanceCustomStore } = await import(
  "../src/features/settings/stores/appearance-custom-store.ts"
);

function otherWindowWrites(change: Record<string, unknown>, key = KEY) {
  const saved = JSON.parse(data.get(key) ?? "{}");
  const newValue = JSON.stringify({
    ...saved,
    state: { ...saved.state, ...change },
  });
  data.set(key, newValue);
  for (const listener of listeners) listener({ key, newValue });
}

test("a name set in another window is not overwritten by an edit in this one", async () => {
  useUserProfileStore.getState().setDisplayName("Old");
  otherWindowWrites({ displayName: "Alice", greetingText: "Hi Alice" });
  await Promise.resolve();
  assert.equal(useUserProfileStore.getState().displayName, "Alice");
  assert.equal(useUserProfileStore.getState().greetingText, "Hi Alice");
  useUserProfileStore.getState().setNickname("Al");
  const stored = JSON.parse(data.get(KEY) ?? "{}").state;
  assert.equal(stored.displayName, "Alice");
  assert.equal(stored.greetingText, "Hi Alice");
  assert.equal(stored.nickname, "Al");
});

test("an appearance change made in another window is not overwritten by an edit in this one", async () => {
  const store = useAppearanceCustomStore;
  store.getState().patch({ uiFont: "Old Font" });
  const saved = JSON.parse(data.get(APPEARANCE_KEY) ?? "{}").state;
  otherWindowWrites(
    {
      customization: {
        ...saved.customization,
        uiFont: "Other Font",
        colors: {
          ...saved.customization.colors,
          light: { ...saved.customization.colors.light, accent: "#ff0000" },
        },
      },
    },
    APPEARANCE_KEY,
  );
  await Promise.resolve();
  assert.equal(store.getState().customization.uiFont, "Other Font");
  assert.equal(store.getState().customization.colors.light.accent, "#ff0000");
  store.getState().patch({ headingFont: "Heading" });
  const stored = JSON.parse(data.get(APPEARANCE_KEY) ?? "{}").state;
  assert.equal(stored.customization.uiFont, "Other Font");
  assert.equal(stored.customization.colors.light.accent, "#ff0000");
  assert.equal(stored.customization.headingFont, "Heading");
});

test("another window resetting local preferences keeps this window's appearance", async () => {
  const store = useAppearanceCustomStore;
  store.getState().patch({ uiFont: "Kept Font" });
  data.delete(APPEARANCE_KEY);
  for (const listener of listeners)
    listener({ key: APPEARANCE_KEY, newValue: null });
  await Promise.resolve();
  assert.equal(store.getState().customization.uiFont, "Kept Font");
});
