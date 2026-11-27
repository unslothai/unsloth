// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
const { useChatPreferencesStore: preferences } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

test("the chat layout defaults to left-to-right and persists the chosen direction", async () => {
  assert.equal(preferences.getInitialState().rightToLeftLayout, false);
  preferences.getState().setRightToLeftLayout(true);
  const saved = store.get("unsloth_chat_preferences");
  assert.ok(saved);
  preferences.setState(preferences.getInitialState(), true);
  store.set("unsloth_chat_preferences", saved);
  await preferences.persist.rehydrate();
  assert.equal(preferences.getState().rightToLeftLayout, true);
  preferences.getState().setRightToLeftLayout(false);
  assert.equal(preferences.getState().rightToLeftLayout, false);
});

test("older and invalid saved preferences keep the left-to-right default", async () => {
  for (const value of [undefined, "rtl", 1, null]) {
    preferences.setState(preferences.getInitialState(), true);
    store.set(
      "unsloth_chat_preferences",
      JSON.stringify({ state: { rightToLeftLayout: value }, version: 1 }),
    );
    await preferences.persist.rehydrate();
    assert.equal(preferences.getState().rightToLeftLayout, false);
  }
});
