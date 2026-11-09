// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A first turn runs under "__default" until initialize() adopts it. A call that parks after that must
// be filed under the chat, or its row keeps the spinner instead of the approval mark.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);
await import("./helpers/store-stubs/settings-http.ts");
await import("./helpers/store-stubs/chat-history-storage.ts");

const { useChatRuntimeStore } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

test("a run adopted before its call parks resolves to the chat's key", () => {
  const owner = () => {};
  useChatRuntimeStore.getState().setThreadRunning("__default", true, { owner });
  useChatRuntimeStore.getState().adoptDefaultThreadRun("chat-late");
  assert.equal(
    useChatRuntimeStore.getState().runKeyForOwner("__default", owner),
    "chat-late",
  );
  useChatRuntimeStore
    .getState()
    .setThreadRunning("chat-late", false, { owner });
});

test("the run registers under the owner the adapter later looks its key up by", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /const confirmationThreadKey = liveThreadKey\(serverCancel\);/,
  );
  assert.match(
    adapter,
    /runtime\.setThreadRunning\(threadKey, true, \{\s*local: !isExternalRequest,\s*owner: serverCancel,\s*\}\);/,
  );
});
