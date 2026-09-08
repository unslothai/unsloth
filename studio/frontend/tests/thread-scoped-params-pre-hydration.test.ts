// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The settings sheet is live while the initial /api/chat/settings request is still out:
// its sliders and system prompt editor call setParams directly, with no hydration guard
// (only the preset buttons have one). The pairing for the open chat has already begun,
// which is what holds a pill toggled in that window for the chat.
//
// Sampling has to be held the same way. Gating the capture on settingsHydrated, as the
// global HTTP write is, left the edit uncaptured: the chat's own snapshot then applied
// over it, and the pairing capture read it off the store as an installation default, which
// pins it onto the next snapshot-less chat. Drives the real store through that order.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
// Skip the legacy import path: it would look for settings this test never wrote.
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { useChatRuntimeStore, beginThreadScopedPairing } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

const SAVED_CHAT = "thread-with-a-snapshot";
const OTHER_CHAT = "thread-with-no-snapshot";

/** What the installation was set to before the user touched anything. */
const INSTALLATION_TEMPERATURE = 0.7;
/** What the user drags the slider to while the settings request is still out. */
const EDITED_TEMPERATURE = 0.15;
/** What this chat had stored from an earlier visit. */
const STORED_TEMPERATURE = 1.3;

test("a sampling edit made before hydration belongs to the open chat", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: INSTALLATION_TEMPERATURE },
  };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  // The composer is up on a saved chat whose read has not gone out: pairing window open.
  useChatRuntimeStore.getState().setActiveThreadId(SAVED_CHAT);
  beginThreadScopedPairing(SAVED_CHAT);

  // The user opens Advanced settings and drags Temperature; no hydration guard there.
  const before = useChatRuntimeStore.getState().params;
  useChatRuntimeStore
    .getState()
    .setParams({ ...before, temperature: EDITED_TEMPERATURE });
  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    EDITED_TEMPERATURE,
  );

  settingsHttp.release?.();
  await hydrating;

  // The mutation-version fence already kept the response off the key the user moved.
  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    EDITED_TEMPERATURE,
    "hydration overwrote the edit",
  );

  // Hydration done, the chat's own read answers with what it had stored.
  useChatRuntimeStore
    .getState()
    .applyThreadScopedSettings(SAVED_CHAT, { temperature: STORED_TEMPERATURE });
  const inTheEditedChat = useChatRuntimeStore.getState().params.temperature;

  // Leaving for a chat with no snapshot: it must run on the installation value, not on the
  // previous chat's edit, and being pinned on open it keeps whatever it lands on for good.
  useChatRuntimeStore.getState().applyThreadScopedSettings(null, null);
  useChatRuntimeStore.getState().setActiveThreadId(OTHER_CHAT);
  beginThreadScopedPairing(OTHER_CHAT);
  useChatRuntimeStore.getState().applyThreadScopedSettings(OTHER_CHAT, null);
  const inTheNextChat = useChatRuntimeStore.getState().params.temperature;

  // Both together: an uncaptured edit loses the user's value in the chat they made
  // it in AND stands in as the installation default for the next chat.
  assert.deepEqual(
    { inTheEditedChat, inTheNextChat },
    {
      inTheEditedChat: EDITED_TEMPERATURE,
      inTheNextChat: INSTALLATION_TEMPERATURE,
    },
  );
});
