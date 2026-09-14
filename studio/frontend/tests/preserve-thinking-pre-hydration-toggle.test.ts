// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A click made while the settings GET is still out. The scalar mutation fence already
// keeps it on screen, so the value the resolver hands the next model load has to follow
// the same fence: recording the hydrated value over a newer click leaves the switch
// visibly on and silently resolves off at the next load. Own file, and one case: the
// window exists only until the first hydration resolves, and the recorded preference is
// module state a sibling test would have already set.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
// Skip the legacy import path: it would look for settings this test never wrote.
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

  // The store's own fence: the click advanced the mutation version, so hydration
  // leaves the visible switch alone.
  assert.equal(useChatRuntimeStore.getState().preserveThinking, true);
  // And the next model load has to agree with what the user is looking at, whatever
  // the family default says.
  assert.equal(
    resolvePreserveThinkingOnLoad({
      supports_preserve_thinking: true,
      preserve_thinking_default: false,
    }),
    true,
  );
});
