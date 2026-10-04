// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The pairing window from the model side. An edit made while a saved chat's snapshot read
// is still out lives in heldThreadScopedEdits, not in a snapshot, and
// threadScopedSettingsThreadId is still null because nothing has been applied yet.
//
// Every outgoing-model snapshot runs through withoutActiveThreadParams, which used to
// return early on that null id alone. A model switch inside the window therefore
// snapshotted the chat's sampling and prompt into the OUTGOING model's memory, shared with
// every other chat: the next chat opened on that model replays the first chat's prompt and
// sliders. Drives the real store through that order.

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

const QWEN = "unsloth/Qwen3.5-9B-GGUF";
const LLAMA = "unsloth/Llama-4-8B";
const CHAT_A = "chat-a-read-still-out";

/** What the installation is holding before anything happens. */
const INSTALLATION_TEMPERATURE = 0.6;
const INSTALLATION_PROMPT = "INSTALLATION PROMPT";
/** Chat A's sentinels. Either one appearing anywhere shared is the leak. */
const EDITED_TEMPERATURE = 1.37;
const EDITED_PROMPT = "CHAT A ONLY 5f3a";

interface Store {
  useChatRuntimeStore: {
    getState: () => Record<string, (...args: never[]) => unknown> & {
      params: Record<string, unknown>;
      paramsByModel: Record<string, Record<string, unknown>>;
    };
  };
  beginThreadScopedPairing: (threadId: string) => void;
}

/** A scenario-scoped copy of the store: the whole feature lives in module state. */
async function freshStore(scenario: string): Promise<Store> {
  settingsHttp.settings = {
    rememberParamsPerModel: true,
    inferenceParams: {
      temperature: INSTALLATION_TEMPERATURE,
      systemPrompt: INSTALLATION_PROMPT,
    },
  };
  settingsHttp.puts.length = 0;
  const mod = (await import(`${STORE_URL}?scenario=${scenario}`)) as never;
  return mod as Store;
}

test("an edit held for an unpaired chat stays out of the outgoing model's memory", async () => {
  const { useChatRuntimeStore, beginThreadScopedPairing } =
    await freshStore("held-edit-model-memory");
  const state = () => useChatRuntimeStore.getState();
  await state().hydratePersistedSettings();
  state().setCheckpoint(QWEN as never, null as never);

  // A saved chat is on screen, its read unanswered: the pairing window is open.
  state().setActiveThreadId(CHAT_A as never);
  beginThreadScopedPairing(CHAT_A);

  // The user drags temperature and rewrites the prompt: both held for chat A.
  state().setParams({
    ...state().params,
    temperature: EDITED_TEMPERATURE,
    systemPrompt: EDITED_PROMPT,
  } as never);
  assert.equal(state().params.temperature, EDITED_TEMPERATURE);
  assert.equal(state().params.systemPrompt, EDITED_PROMPT);

  // Then the model is switched before the read lands, which snapshots the outgoing one.
  state().setCheckpoint(LLAMA as never, null as never);

  const remembered = state().paramsByModel[QWEN] ?? {};
  assert.notEqual(
    remembered.temperature,
    EDITED_TEMPERATURE,
    "chat A's temperature was remembered against the model it was switched off",
  );
  assert.notEqual(
    remembered.systemPrompt,
    EDITED_PROMPT,
    "chat A's system prompt was remembered against the model it was switched off",
  );
  // What the model is owed is what the installation had, not nothing: a model that was
  // never edited still keeps what it ran with, which is the point of the snapshot.
  assert.equal(remembered.temperature, INSTALLATION_TEMPERATURE);
  assert.equal(remembered.systemPrompt, INSTALLATION_PROMPT);

  // Nor may it reach the installation, which every snapshot-less chat follows.
  const puts = JSON.stringify(settingsHttp.puts);
  assert.ok(!puts.includes(EDITED_PROMPT), "chat A's prompt reached a PUT");
  assert.ok(
    !puts.includes(String(EDITED_TEMPERATURE)),
    "chat A's temperature reached a PUT",
  );
});
