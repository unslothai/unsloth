// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sliders call setParams before hydration; capture must not be gated on settingsHydrated,
// or the edit becomes the next snapshot-less chat's default.

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

const INSTALLATION_TEMPERATURE = 0.7;
const EDITED_TEMPERATURE = 0.15;
const STORED_TEMPERATURE = 1.3;

test("a sampling edit made before hydration belongs to the open chat", async () => {
  settingsHttp.settings = {
    inferenceParams: { temperature: INSTALLATION_TEMPERATURE },
  };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  useChatRuntimeStore.getState().setActiveThreadId(SAVED_CHAT);
  beginThreadScopedPairing(SAVED_CHAT);

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

  assert.equal(
    useChatRuntimeStore.getState().params.temperature,
    EDITED_TEMPERATURE,
    "hydration overwrote the edit",
  );

  useChatRuntimeStore
    .getState()
    .applyThreadScopedSettings(SAVED_CHAT, { temperature: STORED_TEMPERATURE });
  const inTheEditedChat = useChatRuntimeStore.getState().params.temperature;

  useChatRuntimeStore.getState().applyThreadScopedSettings(null, null);
  useChatRuntimeStore.getState().setActiveThreadId(OTHER_CHAT);
  beginThreadScopedPairing(OTHER_CHAT);
  useChatRuntimeStore.getState().applyThreadScopedSettings(OTHER_CHAT, null);
  const inTheNextChat = useChatRuntimeStore.getState().params.temperature;

  assert.deepEqual(
    { inTheEditedChat, inTheNextChat },
    {
      inTheEditedChat: EDITED_TEMPERATURE,
      inTheNextChat: INSTALLATION_TEMPERATURE,
    },
  );
});
