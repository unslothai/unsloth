// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The mechanism the update paths rely on: a preset selection sits in the 400ms settings
// debounce, and flushPendingChatSettings is what puts it on the wire before a restart.

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

test("a preset selection is unsent until the flush, and lands on it", async () => {
  settingsHttp.settings = {
    activePreset: "Default",
    activePresetSource: "builtin-default",
  };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  await flushPendingChatSettings();
  settingsHttp.puts.length = 0;

  useChatRuntimeStore.getState().setActivePreset("Long context");

  // The debounce has not fired: this is the window a backend restart would close over.
  assert.equal(
    settingsHttp.puts.find((put) => put.activePreset !== undefined),
    undefined,
    "the selection reached the server without a flush",
  );

  await flushPendingChatSettings();

  const put = settingsHttp.puts.find((p) => p.activePreset !== undefined);
  assert.equal(put?.activePreset, "Long context");
});

test("a flush with nothing queued sends nothing and arms no timeout", async () => {
  await flushPendingChatSettings();
  settingsHttp.puts.length = 0;

  // The behaviour under test is that the idle path returns before it can arm
  // SETTINGS_FLUSH_TIMEOUT_MS, so watch the timer instead of the wall clock.
  const realSetTimeout = globalThis.setTimeout;
  const delays: number[] = [];
  globalThis.setTimeout = ((fn: never, delay?: number, ...rest: never[]) => {
    delays.push(delay ?? 0);
    return realSetTimeout(fn, delay, ...rest);
  }) as typeof globalThis.setTimeout;
  try {
    await flushPendingChatSettings();
  } finally {
    globalThis.setTimeout = realSetTimeout;
  }

  assert.equal(settingsHttp.puts.length, 0);
  assert.deepEqual(delays, [], "the idle flush armed a timer");
});
