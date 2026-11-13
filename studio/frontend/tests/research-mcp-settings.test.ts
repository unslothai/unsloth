// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";
import { settingsHttp } from "./helpers/store-stubs/settings-http.ts";
import { snapshotQueuedChatRunSettings } from "../src/features/chat/utils/queued-chat-run-settings.ts";

const { store: storage } = installLocalStorageFake();
storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);
const { useChatRuntimeStore, flushPendingChatSettings } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);
const original = [{ serverId: "notes", tool: "search_notes" }];
const later = [{ serverId: "papers", tool: "search_papers" }];

test("MCP sources hydrate from the account and saves reach another session", async () => {
  settingsHttp.settings = { researchMcpSources: original };
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.deepEqual(useChatRuntimeStore.getState().researchMcpSources, original);
  useChatRuntimeStore.getState().setResearchMcpSources(later);
  await flushPendingChatSettings();
  assert.deepEqual(settingsHttp.puts.at(-1)?.researchMcpSources, later);
  useChatRuntimeStore.getState().setResearchMcpSources([]);
  await flushPendingChatSettings();
  assert.deepEqual(settingsHttp.puts.at(-1)?.researchMcpSources, []);
});

test("queued research retains the MCP sources selected when sent", () => {
  useChatRuntimeStore.getState().setResearchMcpSources(original);
  const queued = snapshotQueuedChatRunSettings(useChatRuntimeStore.getState());
  useChatRuntimeStore.getState().setResearchMcpSources(later);
  const runtime = { ...useChatRuntimeStore.getState(), ...queued };
  assert.deepEqual(runtime.researchMcpSources, original);
});
