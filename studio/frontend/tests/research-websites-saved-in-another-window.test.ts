// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: localStorageFake, fireWindowEvent } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./store-settings-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const {
  CROSS_WINDOW_EDIT_KEY,
  CHAT_DEEP_RESEARCH_MCP_SOURCES_KEY,
  CHAT_DEEP_RESEARCH_MODEL_TIMEOUT_KEY,
  CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY,
  useChatRuntimeStore,
} = await import("../src/features/chat/stores/chat-runtime-store.ts");

async function settled(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 600));
}

function writeFromAnotherWindow(key: string, value: unknown): void {
  const raw = typeof value === "number" ? String(value) : JSON.stringify(value);
  localStorageFake.set(key, raw);
  const newValue = JSON.stringify({ key, raw, nonce: Math.random() });
  localStorageFake.set(CROSS_WINDOW_EDIT_KEY, newValue);
  fireWindowEvent("storage", { key: CROSS_WINDOW_EDIT_KEY, newValue });
}

test("a site blocked in another window stays blocked when this window saves its list", async () => {
  settingsHttp.settings = {};
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  assert.deepEqual(useChatRuntimeStore.getState().researchWebsitePolicy, {
    allowedDomains: [],
    blockedDomains: [],
  });

  writeFromAnotherWindow(CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY, {
    allowedDomains: ["arxiv.org"],
    blockedDomains: ["reddit.com"],
  });
  const policy = useChatRuntimeStore.getState().researchWebsitePolicy;
  assert.deepEqual(policy, {
    allowedDomains: ["arxiv.org"],
    blockedDomains: ["reddit.com"],
  });

  useChatRuntimeStore.getState().setResearchWebsitePolicy({
    ...policy,
    blockedDomains: [...policy.blockedDomains, "pinterest.com"],
  });
  await settled();

  const edit = JSON.parse(localStorageFake.get(CROSS_WINDOW_EDIT_KEY) ?? "null");
  assert.equal(edit?.key, CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY);
  assert.equal(edit?.raw, localStorageFake.get(CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY));

  const sent = settingsHttp.puts.filter((put) => "researchWebsitePolicy" in put);
  assert.deepEqual(sent.at(-1)?.researchWebsitePolicy, {
    allowedDomains: ["arxiv.org"],
    blockedDomains: ["reddit.com", "pinterest.com"],
  });
});

test("research sources added in another window show up in this one", () => {
  writeFromAnotherWindow(CHAT_DEEP_RESEARCH_MCP_SOURCES_KEY, [
    { serverId: "docs", tool: "search" },
  ]);
  assert.deepEqual(useChatRuntimeStore.getState().researchMcpSources, [
    { serverId: "docs", tool: "search" },
  ]);
});

test("a request time limit set in another window shows up in this one", () => {
  writeFromAnotherWindow(CHAT_DEEP_RESEARCH_MODEL_TIMEOUT_KEY, 600);
  assert.equal(useChatRuntimeStore.getState().researchModelTimeoutSeconds, 600);
});

test("saving an unchanged list is not announced to other windows as an edit", () => {
  const before = localStorageFake.get(CROSS_WINDOW_EDIT_KEY);
  const { researchWebsitePolicy, setResearchWebsitePolicy } = useChatRuntimeStore.getState();
  setResearchWebsitePolicy({ ...researchWebsitePolicy });
  assert.equal(localStorageFake.get(CROSS_WINDOW_EDIT_KEY), before);
});
