// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Closing the pairing window restores the fields whose edits were held, because those
// edits belong to the chat and must not become the installation default that every
// snapshot-less chat follows. The value it restores is the sample taken when the window
// opened.
//
// That sample goes stale if a model finishes loading inside the window: setParams
// publishes the new model's recommendation, marked fromModelDefaults, so it is NOT
// captured as a chat edit and IS written to /api/chat/settings. Restoring the pre-window
// sample over it leaves this session's in-memory defaults behind the server's, pinning the
// old value onto the next snapshot-less chat until reload. Drives that order.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
// Skip the legacy import path: it would look for settings this test never wrote.
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");

const STORE_URL = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;

const MODEL_A = "unsloth/Model-A";
const MODEL_B = "unsloth/Model-B";
const SAVED_CHAT = "chat-read-still-out";
const SNAPSHOT_LESS_CHAT = "chat-with-no-snapshot";

const INSTALLATION_TEMPERATURE = 0.6;
/** What the user drags the slider to while the chat's read is still out. */
const EDITED_TEMPERATURE = 1.37;
/** What model B recommends when it finishes loading inside that window. */
const B_DEFAULT_TEMPERATURE = 0.31;
/** What the chat turns out to have had stored all along. */
const STORED_TEMPERATURE = 0.9;

/** The debounced settings PUT, plus the write chains it hangs off. */
async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 900));
}

test("a default published inside the pairing window survives the window closing", async () => {
  settingsHttp.settings = {
    // Off, so nothing here is about a model's own remembered entry.
    rememberParamsPerModel: false,
    inferenceParams: { temperature: INSTALLATION_TEMPERATURE },
  };
  settingsHttp.puts.length = 0;
  const { useChatRuntimeStore, beginThreadScopedPairing } = (await import(
    `${STORE_URL}?scenario=pairing-default-refresh`
  )) as never as {
    useChatRuntimeStore: {
      getState: () => Record<string, (...args: never[]) => unknown> & {
        params: Record<string, unknown>;
      };
    };
    beginThreadScopedPairing: (threadId: string) => void;
  };
  const state = () => useChatRuntimeStore.getState();
  await state().hydratePersistedSettings();
  state().setCheckpoint(MODEL_A as never, null as never);

  // A saved chat is on screen and its read has not answered yet.
  state().setActiveThreadId(SAVED_CHAT as never);
  beginThreadScopedPairing(SAVED_CHAT);

  // The user drags temperature: the edit is held for this chat.
  state().setParams({
    ...state().params,
    temperature: EDITED_TEMPERATURE,
  } as never);

  // Model B finishes loading in the same window and publishes its recommendation.
  state().setParams(
    {
      ...state().params,
      checkpoint: MODEL_B,
      temperature: B_DEFAULT_TEMPERATURE,
    } as never,
    { fromModelDefaults: true } as never,
  );
  await settle();

  // That value went to the installation, so it is what the defaults now are.
  const sentTemperatures = settingsHttp.puts
    .map((put) => (put.inferenceParams as Record<string, unknown>)?.temperature)
    .filter((value) => value !== undefined);
  assert.deepEqual(
    sentTemperatures,
    [B_DEFAULT_TEMPERATURE],
    "the model default published inside the window never reached the installation",
  );

  // The read finally lands and the window closes.
  state().applyThreadScopedSettings(SAVED_CHAT as never, {
    temperature: STORED_TEMPERATURE,
  } as never);

  // A chat with no snapshot of its own follows the installation defaults, which the
  // server holds as B's value. Reverting to the pre-window sample here is the bug.
  state().setActiveThreadId(SNAPSHOT_LESS_CHAT as never);
  beginThreadScopedPairing(SNAPSHOT_LESS_CHAT);
  state().applyThreadScopedSettings(SNAPSHOT_LESS_CHAT as never, null as never);
  assert.equal(
    state().params.temperature,
    B_DEFAULT_TEMPERATURE,
    "a snapshot-less chat is pinned with the pre-window default the server no longer holds",
  );
});
