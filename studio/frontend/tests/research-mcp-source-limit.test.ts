// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
const sources = Array.from({ length: 21 }, (_, index) => ({
  serverId: `server-${index}`,
  tool: `search-${index}`,
}));
localStorageFake.set(
  "unsloth_chat_deep_research_mcp_sources",
  JSON.stringify([...sources, sources[0], { serverId: "", tool: "search" }]),
);
register("./store-settings-resolver.mjs", import.meta.url);

const { MAX_RESEARCH_MCP_SOURCES, useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

test("MCP research sources stay within the backend request limit", () => {
  assert.equal(MAX_RESEARCH_MCP_SOURCES, 20);
  assert.deepEqual(
    useChatRuntimeStore.getState().researchMcpSources,
    sources.slice(0, MAX_RESEARCH_MCP_SOURCES),
  );

  const settingsEpoch = useChatRuntimeStore.getState().queuedSettingsEpoch;
  useChatRuntimeStore
    .getState()
    .setResearchMcpSources([
      ...sources,
      sources[0],
      { serverId: "server", tool: "" },
    ]);
  assert.deepEqual(
    useChatRuntimeStore.getState().researchMcpSources,
    sources.slice(0, MAX_RESEARCH_MCP_SOURCES),
  );
  assert.equal(
    useChatRuntimeStore.getState().queuedSettingsEpoch,
    settingsEpoch + 1,
  );
  assert.deepEqual(
    JSON.parse(
      localStorageFake.get("unsloth_chat_deep_research_mcp_sources") || "[]",
    ),
    sources.slice(0, MAX_RESEARCH_MCP_SOURCES),
  );
});
