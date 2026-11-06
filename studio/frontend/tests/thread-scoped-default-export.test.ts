// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat exports read threadScopedDefault for a chat whose snapshot omits the system prompt. While
// the first chat's read is out, an edit to its prompt is held in the live store, and that edit
// must not become the default every other snapshot-less chat exports.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");

const STORE_URL = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;

const INSTALLATION_PROMPT = "INSTALLATION PROMPT";
const EDITED_PROMPT = "CHAT A ONLY 9c1e";

test("a prompt edit held for a pairing chat is not the default other chats export", async () => {
  settingsHttp.settings = { inferenceParams: { systemPrompt: INSTALLATION_PROMPT } };
  settingsHttp.puts.length = 0;
  const { useChatRuntimeStore, beginThreadScopedPairing, threadScopedDefault } =
    (await import(`${STORE_URL}?scenario=default-export`)) as never as {
      useChatRuntimeStore: {
        getState: () => Record<string, (...args: never[]) => unknown> & {
          params: Record<string, unknown>;
        };
      };
      beginThreadScopedPairing: (threadId: string) => void;
      threadScopedDefault: (key: string) => unknown;
    };
  const state = () => useChatRuntimeStore.getState();
  await state().hydratePersistedSettings();
  assert.equal(threadScopedDefault("systemPrompt"), INSTALLATION_PROMPT);

  state().setActiveThreadId("chat-a" as never);
  beginThreadScopedPairing("chat-a");
  state().setParams({ ...state().params, systemPrompt: EDITED_PROMPT } as never);
  assert.equal(state().params.systemPrompt, EDITED_PROMPT);

  assert.equal(threadScopedDefault("systemPrompt"), INSTALLATION_PROMPT);
});
