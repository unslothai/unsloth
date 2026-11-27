// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The resolver must follow the scalar mutation fence. Own file, one case: the preference is
// module state.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { resolvePreserveThinkingOnLoad, useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

test("a toggle made before hydration lands outlives the value it overtook", async () => {
  settingsHttp.settings = { preserveThinking: false };
  settingsHttp.hold();
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();
  useChatRuntimeStore.getState().setPreserveThinking(true);
  settingsHttp.release?.();
  await hydrating;

  assert.equal(useChatRuntimeStore.getState().preserveThinking, true);
  assert.equal(
    resolvePreserveThinkingOnLoad({
      supports_preserve_thinking: true,
      preserve_thinking_default: false,
    }),
    true,
  );
});
