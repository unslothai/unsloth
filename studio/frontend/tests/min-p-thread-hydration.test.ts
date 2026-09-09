// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: storage } = installLocalStorageFake();
storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);
const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);
const {
  useChatRuntimeStore: store,
  beginThreadScopedPairing,
  flushPendingChatSettings,
  awaitStartedThreadScopedSettingsWrites,
} = await import("../src/features/chat/stores/chat-runtime-store.ts");

test("a held Min P choice survives both delayed reads, persists to its thread and stays out of defaults", async () => {
  settingsHttp.settings = {
    inferenceParams: { minP: 0.2, minPMode: "server-default" },
  };
  settingsHttp.hold();
  const hydrating = store.getState().hydratePersistedSettings();
  store.getState().setActiveThreadId("edited");
  beginThreadScopedPairing("edited");
  store
    .getState()
    .setParams(
      { ...store.getState().params, minP: 0, minPMode: "custom" },
      { minPChoiceEdited: true },
    );
  settingsHttp.release?.();
  await hydrating;
  store.getState().applyThreadScopedSettings("edited", { minP: 0.7 });
  assert.equal(store.getState().params.minP, 0);
  assert.equal(store.getState().params.minPMode, "custom");

  // Leaving flushes the original thread's normal settings write.
  store.getState().applyThreadScopedSettings(null, null);
  await awaitStartedThreadScopedSettingsWrites();
  await flushPendingChatSettings();
  const saved = threadRows.rows.get("edited");
  assert.equal(saved?.minP, 0);
  assert.equal(saved?.minPMode, "custom");
  assert.equal(store.getState().params.minP, 0.2);
  assert.equal(store.getState().params.minPMode, "server-default");
  for (const put of settingsHttp.puts) {
    const global = put.inferenceParams as Record<string, unknown> | undefined;
    assert.notEqual(global?.minPMode, "custom");
    assert.notEqual(global?.minP, 0);
  }

  store.getState().setActiveThreadId("other");
  beginThreadScopedPairing("other");
  store.getState().applyThreadScopedSettings("other", {});
  assert.equal(store.getState().params.minPMode, "server-default");
  store.getState().applyThreadScopedSettings(null, null);
  store.getState().setActiveThreadId("edited");
  beginThreadScopedPairing("edited");
  store.getState().applyThreadScopedSettings("edited", saved);
  assert.equal(store.getState().params.minP, 0);
  assert.equal(store.getState().params.minPMode, "custom");
  store.getState().applyThreadScopedSettings(null, null);
});

test("recommendations retain installation mode beneath an active thread override", async () => {
  store.getState().setActiveThreadId(null);
  store
    .getState()
    .setParams({
      ...store.getState().params,
      minP: 0.2,
      minPMode: "server-default",
    });
  store.getState().setActiveThreadId("custom-thread");
  beginThreadScopedPairing("custom-thread");
  store
    .getState()
    .applyThreadScopedSettings("custom-thread", {
      minP: 0,
      minPMode: "custom",
    });
  store
    .getState()
    .setParams(
      { ...store.getState().params, minP: 0.4 },
      { fromModelDefaults: true },
    );
  assert.equal(store.getState().params.minP, 0);
  assert.equal(store.getState().params.minPMode, "custom");
  store.getState().applyThreadScopedSettings(null, null);
  assert.equal(store.getState().params.minP, 0.4);
  assert.equal(store.getState().params.minPMode, "server-default");
  await awaitStartedThreadScopedSettingsWrites();
});
