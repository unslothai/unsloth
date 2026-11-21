// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A model loading inside the pairing window publishes new defaults to the server; restoring
// the pre-window sample would leave in-memory defaults stale.

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
const EDITED_TEMPERATURE = 1.37;
const B_DEFAULT_TEMPERATURE = 0.31;
const STORED_TEMPERATURE = 0.9;

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 900));
}

test("a default published inside the pairing window survives the window closing", async () => {
  settingsHttp.settings = {
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

  state().setActiveThreadId(SAVED_CHAT as never);
  beginThreadScopedPairing(SAVED_CHAT);

  state().setParams({
    ...state().params,
    temperature: EDITED_TEMPERATURE,
  } as never);

  state().setParams(
    {
      ...state().params,
      checkpoint: MODEL_B,
      temperature: B_DEFAULT_TEMPERATURE,
    } as never,
    { fromModelDefaults: true } as never,
  );
  await settle();

  const sentTemperatures = settingsHttp.puts
    .map((put) => (put.inferenceParams as Record<string, unknown>)?.temperature)
    .filter((value) => value !== undefined);
  assert.deepEqual(
    sentTemperatures,
    [B_DEFAULT_TEMPERATURE],
    "the model default published inside the window never reached the installation",
  );

  state().applyThreadScopedSettings(SAVED_CHAT as never, {
    temperature: STORED_TEMPERATURE,
  } as never);

  // A snapshot-less chat follows the installation defaults, which now hold B's value.
  state().setActiveThreadId(SNAPSHOT_LESS_CHAT as never);
  beginThreadScopedPairing(SNAPSHOT_LESS_CHAT);
  state().applyThreadScopedSettings(SNAPSHOT_LESS_CHAT as never, null as never);
  assert.equal(
    state().params.temperature,
    B_DEFAULT_TEMPERATURE,
    "a snapshot-less chat is pinned with the pre-window default the server no longer holds",
  );
});
