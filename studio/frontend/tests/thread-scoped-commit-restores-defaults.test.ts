// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Leaving a chat mid-read must restore installation values over its edits, or the next chat
// captures them. Retry and fork commit with the chat still open and must keep the edit.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
// Skip the legacy import path: it would look for settings this test never wrote.
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);

const STORE_URL = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;

const MODEL_A = "unsloth/Model-A";
const CHAT_A = "chat-whose-read-is-still-out";
const CHAT_B = "chat-with-no-snapshot";

const INSTALLATION_TEMPERATURE = 0.6;
const EDITED_TEMPERATURE = 1.37;
const EDITED_PROMPT = "answer only in haiku";

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 900));
}

interface StoreModule {
  useChatRuntimeStore: {
    getState: () => Record<string, (...args: never[]) => unknown> & {
      params: Record<string, unknown>;
    };
  };
  beginThreadScopedPairing: (threadId: string) => void;
  commitHeldThreadScopedEditsToTheirThread: () => Promise<void>;
}

async function bootStore(scenario: string): Promise<StoreModule> {
  settingsHttp.settings = {
    rememberParamsPerModel: false,
    inferenceParams: { temperature: INSTALLATION_TEMPERATURE },
  };
  settingsHttp.puts.length = 0;
  threadRows.reset();
  const mod = (await import(
    `${STORE_URL}?scenario=${scenario}`
  )) as never as StoreModule;
  await mod.useChatRuntimeStore.getState().hydratePersistedSettings();
  mod.useChatRuntimeStore
    .getState()
    .setCheckpoint(MODEL_A as never, null as never);
  return mod;
}

test("a chat left mid-read keeps its edits, and the next chat gets the defaults", async () => {
  const {
    useChatRuntimeStore,
    beginThreadScopedPairing,
    commitHeldThreadScopedEditsToTheirThread,
  } = await bootStore("commit-restores-defaults");
  const state = () => useChatRuntimeStore.getState();

  state().setActiveThreadId(CHAT_A as never);
  beginThreadScopedPairing(CHAT_A);

  state().setParams({
    ...state().params,
    temperature: EDITED_TEMPERATURE,
    systemPrompt: EDITED_PROMPT,
  } as never);

  state().setActiveThreadId(CHAT_B as never);
  await commitHeldThreadScopedEditsToTheirThread();
  await settle();

  assert.deepEqual(
    threadRows.writesFor(CHAT_A).map((write) => write.settingsPatch),
    [{ temperature: EDITED_TEMPERATURE, systemPrompt: EDITED_PROMPT }],
    "the held edits did not reach the chat they were made in",
  );

  beginThreadScopedPairing(CHAT_B);
  state().applyThreadScopedSettings(CHAT_B as never, null as never);
  await settle();

  assert.equal(
    state().params.temperature,
    INSTALLATION_TEMPERATURE,
    "chat A's temperature followed the user into chat B",
  );
  assert.equal(
    state().params.systemPrompt,
    "",
    "chat A's system prompt followed the user into chat B",
  );
  const pinned = threadRows.rows.get(CHAT_B) ?? {};
  assert.equal(
    pinned.temperature,
    INSTALLATION_TEMPERATURE,
    "chat B was pinned with chat A's temperature",
  );
  assert.equal(
    pinned.systemPrompt,
    "",
    "chat B was pinned with chat A's system prompt",
  );
});

test("a commit made with the chat still open leaves the edit on screen", async () => {
  const {
    useChatRuntimeStore,
    beginThreadScopedPairing,
    commitHeldThreadScopedEditsToTheirThread,
  } = await bootStore("commit-keeps-open-chat");
  const state = () => useChatRuntimeStore.getState();

  state().setActiveThreadId(CHAT_A as never);
  beginThreadScopedPairing(CHAT_A);
  state().setParams({
    ...state().params,
    temperature: EDITED_TEMPERATURE,
  } as never);

  await commitHeldThreadScopedEditsToTheirThread();
  await settle();

  assert.equal(
    state().params.temperature,
    EDITED_TEMPERATURE,
    "the retry commit put the defaults back over an edit the user is looking at",
  );
});
