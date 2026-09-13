// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An update accepted while the initial settings GET is still out must not push this session's
// values over the stored row. Own file: the store is a module singleton, so holding hydration
// from the very first call needs a process where nothing has hydrated it yet.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { flushPendingChatSettings, useChatRuntimeStore } =
  await import("../src/features/chat/stores/chat-runtime-store.ts");

test("a flush while hydration is in flight leaves the held edit unsent", async () => {
  settingsHttp.hold();
  settingsHttp.puts.length = 0;
  const hydration = useChatRuntimeStore.getState().hydratePersistedSettings();

  useChatRuntimeStore.getState().setToolsEnabled(false);
  await flushPendingChatSettings();

  assert.equal(
    settingsHttp.puts.length,
    0,
    "the flush sent a pre-hydration edit before the GET landed",
  );

  settingsHttp.release?.();
  await hydration;

  // And the edit is not dropped: hydration replays it onto the outgoing patch.
  await flushPendingChatSettings();
  assert.ok(
    settingsHttp.puts.some((put) => put.toolsEnabled === false),
    "the held edit was lost",
  );
});
