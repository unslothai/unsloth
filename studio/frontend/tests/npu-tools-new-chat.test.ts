// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { type TestContext } from "node:test";
import type * as chatRuntime from "../src/features/chat/stores/chat-runtime-store.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: storage } = installLocalStorageFake();
register("./thread-sampling-resolver.mjs", import.meta.url);
const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import("./helpers/store-stubs/chat-history-storage.ts");
const runtimeUrl = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;
const { applyActiveModelStatusToStore } = await import(
  "../src/features/chat/lib/apply-inference-status-to-store.ts"
);

const GPU = "unsloth/Qwen3-8B";
const NPU = "lemonade:qwen3.8-tk-4b-FLM";

async function newChatAfterLoading(t: TestContext, model: string, saved?: boolean) {
  storage.clear();
  storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
  if (saved !== undefined) storage.set("unsloth_chat_tools_enabled", String(saved));
  settingsHttp.settings = { rememberParamsPerModel: false };
  threadRows.reset();
  const runtime = (await import(
    `${runtimeUrl}?scenario=${encodeURIComponent(t.name)}`
  )) as typeof chatRuntime;
  const store = runtime.useChatRuntimeStore;
  t.after(async () => {
    await runtime.flushPendingChatSettings();
    await runtime.awaitStartedThreadScopedSettingsWrites();
  });
  await store.getState().hydratePersistedSettings();
  store.getState().setCheckpoint(GPU, null);
  store.setState({ supportsTools: true, ...runtime.resolveToolsEnabledOnLoad(true, GPU) });
  // A chat opened under the GPU model captures its Search-on default.
  store.getState().setActiveThreadId("gpu-chat");
  store.getState().applyThreadScopedSettings("gpu-chat", null);
  await runtime.awaitThreadScopedSettingsWrite("gpu-chat");
  store.getState().setCheckpoint(model, null);
  applyActiveModelStatusToStore(
    { active_model: model, model_identifier: model, supports_tools: true } as never,
    { previousCheckpoint: GPU },
  );
  store.getState().setActiveThreadId(null);
  store.getState().applyThreadScopedSettings(null, null);
  const composer = store.getState().toolsEnabled;
  store.getState().setActiveThreadId("new-chat");
  store.getState().applyThreadScopedSettings("new-chat", null);
  await runtime.awaitThreadScopedSettingsWrite("new-chat");
  return {
    composer,
    live: store.getState().toolsEnabled,
    code: store.getState().codeToolsEnabled,
    stored: threadRows.rows.get("new-chat")?.toolsEnabled,
  };
}

test("a new chat on an NPU model starts with Search and Code off", async (t) => {
  assert.deepEqual(await newChatAfterLoading(t, NPU), {
    composer: false,
    live: false,
    code: false,
    stored: false,
  });
});

test("a new chat on a GPU model still starts with Search and Code on", async (t) => {
  assert.deepEqual(await newChatAfterLoading(t, "unsloth/gemma-4-E2B-it"), {
    composer: true,
    live: true,
    code: true,
    stored: true,
  });
});

test("a new chat on an NPU model keeps a saved choice to turn Search on", async (t) => {
  const result = await newChatAfterLoading(t, NPU, true);
  assert.equal(result.live, true);
  assert.equal(result.stored, true);
  assert.equal(result.code, false);
});
