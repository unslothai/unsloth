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
const { CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY, CROSS_WINDOW_EDIT_KEY, useChatRuntimeStore } =
  await import("../src/features/chat/stores/chat-runtime-store.ts");

const KEY = CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY;
const policy = (...blockedDomains: string[]) => ({ allowedDomains: [], blockedDomains });

function editInAnotherWindow(value: unknown): void {
  const raw = JSON.stringify(value);
  localStorageFake.set(KEY, raw);
  const newValue = JSON.stringify({ key: KEY, nonce: Math.random() });
  localStorageFake.set(CROSS_WINDOW_EDIT_KEY, newValue);
  fireWindowEvent("storage", { key: CROSS_WINDOW_EDIT_KEY, newValue });
}

test("a settings load answered before another window's save does not undo that save", async () => {
  let answer!: (settings: Record<string, unknown>) => void;
  settingsHttp.getResponses.push(new Promise((resolve) => (answer = resolve)));
  const hydrating = useChatRuntimeStore.getState().hydratePersistedSettings();

  editInAnotherWindow(policy("reddit.com", "pinterest.com"));
  answer({ researchWebsitePolicy: policy("reddit.com") });
  await hydrating;

  assert.deepEqual(useChatRuntimeStore.getState().researchWebsitePolicy, policy("reddit.com", "pinterest.com"));
  assert.deepEqual(JSON.parse(localStorageFake.get(KEY) ?? "null"), policy("reddit.com", "pinterest.com"));
});

test("another window caching an older server value is not taken as an edit", () => {
  const stale = JSON.stringify(policy("reddit.com"));
  localStorageFake.set(KEY, stale);
  fireWindowEvent("storage", { key: KEY, newValue: stale });
  assert.deepEqual(useChatRuntimeStore.getState().researchWebsitePolicy, policy("reddit.com", "pinterest.com"));
});
