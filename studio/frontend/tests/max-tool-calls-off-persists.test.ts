// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();

const {
  loadChatSettingsWithLegacyImport,
  loadLegacyChatSettings,
  sanitizeChatSettings,
  savePersistedChatSettingsPatch,
} = await import("../src/features/chat/utils/chat-settings-storage.ts");

test("Off survives the outgoing patch", () => {
  assert.deepEqual(sanitizeChatSettings({ maxToolCallsPerMessage: 0 }), {
    maxToolCallsPerMessage: 0,
  });
  assert.deepEqual(sanitizeChatSettings({ maxToolCallsPerMessage: -1 }), {});
  assert.deepEqual(sanitizeChatSettings({ toolCallTimeout: 0 }), {});
});

test("Off survives hydration from localStorage", () => {
  store.set("unsloth_max_tool_calls_per_message", "0");
  store.set("unsloth_tool_call_timeout", "0");
  assert.deepEqual(loadLegacyChatSettings(), { maxToolCallsPerMessage: 0 });
  store.clear();
});

test("Off is sent to the server and read back on reload", async () => {
  const server: Record<string, unknown> = { maxToolCallsPerMessage: 25 };
  const bodies: string[] = [];
  setAuthFetchHandler((_input, init) => {
    if (init?.method === "PUT") {
      bodies.push(String(init.body));
      Object.assign(server, JSON.parse(String(init.body)));
    }
    return Response.json({ settings: server });
  });
  try {
    const saved = await savePersistedChatSettingsPatch({
      maxToolCallsPerMessage: 0,
    });
    assert.deepEqual(bodies, ['{"maxToolCallsPerMessage":0}']);
    assert.equal(saved.maxToolCallsPerMessage, 0);
    const reloaded = await loadChatSettingsWithLegacyImport();
    assert.equal(reloaded.settings.maxToolCallsPerMessage, 0);
  } finally {
    setAuthFetchHandler(null);
  }
});
