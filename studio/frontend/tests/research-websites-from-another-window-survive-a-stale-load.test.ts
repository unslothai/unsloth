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
const { CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY, useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

test("a settings load answered before another window's save does not undo that save", async () => {
  const stale = { allowedDomains: [], blockedDomains: ["reddit.com"] };
  const fresh = { allowedDomains: [], blockedDomains: ["reddit.com", "pinterest.com"] };
  let answer!: (settings: Record<string, unknown>) => void;
  settingsHttp.getResponses.push(new Promise((resolve) => (answer = resolve)));
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  const newValue = JSON.stringify(fresh);
  localStorageFake.set(CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY, newValue);
  fireWindowEvent("storage", { key: CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY, newValue });
  answer({ researchWebsitePolicy: stale });
  await hydrating;

  assert.deepEqual(useChatRuntimeStore.getState().researchWebsitePolicy, fresh);
  assert.deepEqual(JSON.parse(localStorageFake.get(CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY) ?? "null"), fresh);
});
