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
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);
const { useExternalProvidersStore } = await import(
  "../src/features/chat/stores/external-providers-store.ts"
);
type RuntimeModule = typeof chatRuntime;
const runtimeUrl = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;
const MAX = 9999;

async function boot(t: TestContext): Promise<RuntimeModule> {
  storage.clear();
  storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
  settingsHttp.settings = { rememberParamsPerModel: false };
  settingsHttp.puts.length = 0;
  threadRows.reset();
  useExternalProvidersStore.setState({ providers: [] });
  const runtime = (await import(
    `${runtimeUrl}?scenario=${encodeURIComponent(t.name)}`
  )) as RuntimeModule;
  t.after(() => runtime.flushPendingChatSettings());
  const store = runtime.useChatRuntimeStore;
  await store.getState().hydratePersistedSettings();
  store.setState({ supportsTools: true });
  return runtime;
}

function cap(runtime: RuntimeModule): number {
  return runtime.useChatRuntimeStore.getState().maxToolCallsPerMessage;
}

test("the default is 50 tool calls per message", async (t) => {
  const runtime = await boot(t);
  assert.equal(runtime.DEFAULT_MAX_TOOL_CALLS_PER_MESSAGE, 50);
  assert.equal(cap(runtime), 50);
});

test("turning MCP on lifts the cap to Max and turning it off restores it", async (t) => {
  const runtime = await boot(t);
  const state = () => runtime.useChatRuntimeStore.getState();
  state().setMaxToolCallsPerMessage(12);
  state().setMcpEnabledForChat(true);
  assert.equal(cap(runtime), MAX);
  await runtime.flushPendingChatSettings();
  assert.equal(settingsHttp.puts.at(-1)?.maxToolCallsPerMessage, MAX); // saved like a slider move
  state().setMcpEnabledForChat(true); // already on: must not overwrite the saved 12
  state().setMcpEnabledForChat(false);
  assert.equal(cap(runtime), 12);
  await runtime.flushPendingChatSettings();
  assert.equal(settingsHttp.puts.at(-1)?.maxToolCallsPerMessage, 12);
});

test("a value picked while MCP is on survives turning MCP off", async (t) => {
  const runtime = await boot(t);
  const state = () => runtime.useChatRuntimeStore.getState();
  state().setMcpEnabledForChat(true);
  assert.equal(cap(runtime), MAX);
  state().setMaxToolCallsPerMessage(30);
  state().setMcpEnabledForChat(false);
  assert.equal(cap(runtime), 30);
});

test("Off and Max are restored as they were", async (t) => {
  const runtime = await boot(t);
  const state = () => runtime.useChatRuntimeStore.getState();
  state().setMaxToolCallsPerMessage(0);
  state().setMcpEnabledForChat(true);
  assert.equal(cap(runtime), MAX);
  state().setMcpEnabledForChat(false);
  assert.equal(cap(runtime), 0);
  state().setMaxToolCallsPerMessage(MAX);
  state().setMcpEnabledForChat(true);
  state().setMcpEnabledForChat(false);
  assert.equal(cap(runtime), MAX);
});

test("Deep Research switching MCP off restores the cap", async (t) => {
  const runtime = await boot(t);
  const state = () => runtime.useChatRuntimeStore.getState();
  state().setMcpEnabledForChat(true);
  assert.equal(cap(runtime), MAX);
  state().setDeepResearchEnabled(true);
  assert.equal(state().mcpEnabledForChat, false);
  assert.equal(cap(runtime), 50);
});
