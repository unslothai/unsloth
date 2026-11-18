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

test("Max picked by hand while MCP is on stays after turning MCP off", async (t) => {
  const runtime = await boot(t);
  const state = () => runtime.useChatRuntimeStore.getState();
  state().setMaxToolCallsPerMessage(12);
  state().setMcpEnabledForChat(true);
  state().setMaxToolCallsPerMessage(30);
  state().setMaxToolCallsPerMessage(MAX);
  state().setMcpEnabledForChat(false);
  assert.equal(cap(runtime), MAX);
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

test("switching to a chat with MCP off restores the cap, and back lifts it", async (t) => {
  const runtime = await boot(t);
  const state = () => runtime.useChatRuntimeStore.getState();
  state().setMaxToolCallsPerMessage(12);
  state().setActiveThreadId("mcp-chat");
  state().applyThreadScopedSettings("mcp-chat", null);
  await runtime.awaitThreadScopedSettingsWrite("mcp-chat");
  state().setMcpEnabledForChat(true);
  assert.equal(cap(runtime), MAX);
  await runtime.awaitThreadScopedSettingsWrite("mcp-chat");
  state().setActiveThreadId("plain-chat");
  state().applyThreadScopedSettings("plain-chat", {
    mcpEnabledForChat: false,
  } as never);
  assert.equal(state().mcpEnabledForChat, false);
  assert.equal(cap(runtime), 12);
  state().setActiveThreadId("mcp-chat");
  state().applyThreadScopedSettings("mcp-chat", {
    mcpEnabledForChat: true,
  } as never);
  assert.equal(state().mcpEnabledForChat, true);
  assert.equal(cap(runtime), MAX);
});

test("after a reload with MCP's Max saved, a chat with MCP off gets the old cap back", async (t) => {
  const first = await boot(t);
  const before = () => first.useChatRuntimeStore.getState();
  before().setMaxToolCallsPerMessage(12);
  before().setActiveThreadId("mcp-chat");
  before().applyThreadScopedSettings("mcp-chat", null);
  before().setMcpEnabledForChat(true);
  await first.flushPendingChatSettings();
  assert.equal(settingsHttp.puts.at(-1)?.maxToolCallsPerMessage, MAX);

  settingsHttp.settings = { ...settingsHttp.settings, maxToolCallsPerMessage: MAX };
  const reloaded = (await import(
    `${runtimeUrl}?scenario=${encodeURIComponent(`${t.name} reload`)}`
  )) as RuntimeModule;
  t.after(() => reloaded.flushPendingChatSettings());
  const state = () => reloaded.useChatRuntimeStore.getState();
  await state().hydratePersistedSettings();
  state().setActiveThreadId("plain-chat");
  state().applyThreadScopedSettings("plain-chat", {
    mcpEnabledForChat: false,
  } as never);
  assert.equal(cap(reloaded), 12);
  state().setActiveThreadId("mcp-chat");
  state().applyThreadScopedSettings("mcp-chat", {
    mcpEnabledForChat: true,
  } as never);
  assert.equal(cap(reloaded), MAX);
});

test("chats opened before settings hydrate still get the cap reconciled", async (t) => {
  await boot(t);
  for (const [mcpChat, saved] of [
    [false, MAX],
    [true, MAX],
    [true, 12],
  ] as const) {
    // Max on the server with the pre-MCP 12 in this browser, or a plain 12.
    settingsHttp.settings = { rememberParamsPerModel: false, maxToolCallsPerMessage: saved };
    storage.set("unsloth_max_tool_calls_before_mcp", saved === MAX ? "12" : "");
    const reloaded = (await import(
      `${runtimeUrl}?scenario=${encodeURIComponent(`${t.name} ${mcpChat} ${saved}`)}`
    )) as RuntimeModule;
    t.after(() => reloaded.flushPendingChatSettings());
    const state = () => reloaded.useChatRuntimeStore.getState();
    state().setActiveThreadId("chat");
    state().applyThreadScopedSettings("chat", {
      mcpEnabledForChat: mcpChat,
    } as never);
    await state().hydratePersistedSettings();
    assert.equal(cap(reloaded), mcpChat ? MAX : 12);
  }
});
