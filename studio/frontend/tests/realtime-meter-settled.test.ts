// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The realtime context meter steps aside once a pass's own context_usage lands. The mark lives on the
// thread's live progress in the store, so a chat switch, a remount or a pass with no decode frame keeps it.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);
await import("./helpers/store-stubs/settings-http.ts");
await import("./helpers/store-stubs/chat-history-storage.ts");

const { useChatRuntimeStore } = await import("../src/features/chat/stores/chat-runtime-store.ts");
const { liveContextUsage } = await import("../src/features/chat/lib/live-generation-stats.ts");

const prefill = { processed: 5316, total: 5316, cached: 0 } as never;
const decode = { tokens: 12 } as never;
const live = (key: string) => useChatRuntimeStore.getState().liveGenerationByThreadId[key];

test("a pass's own context_usage settles its live progress until the next pass starts", () => {
  const store = useChatRuntimeStore.getState();
  store.setLivePrefill("t1", prefill);
  assert.ok(liveContextUsage(live("t1")), "prefill counts live");
  // A short tool-call pass: its usage lands before any decode frame.
  store.settleLiveGeneration("t1");
  assert.equal(live("t1")?.settled, true);
  assert.equal(liveContextUsage(live("t1")), null, "the exact settled count shows");
  store.setLiveDecode("t1", decode);
  assert.equal(live("t1")?.settled, true, "a trailing decode frame keeps it settled");
  store.settleLiveGeneration("t1");
  store.setLivePrefill("t1", prefill);
  assert.equal(live("t1")?.settled, undefined);
  assert.ok(liveContextUsage(live("t1")));
  store.clearLiveGenerationForThread("t1");
});

test("settling is per thread and survives a remount", () => {
  const store = useChatRuntimeStore.getState();
  store.setLivePrefill("a", prefill);
  store.setLivePrefill("b", prefill);
  store.settleLiveGeneration("a");
  assert.equal(live("a")?.settled, true);
  assert.equal(live("b")?.settled, undefined, "the other chat keeps counting live");
  store.clearLiveGenerationForThread("a");
  const before = useChatRuntimeStore.getState().liveGenerationByThreadId;
  store.settleLiveGeneration("a");
  assert.equal(useChatRuntimeStore.getState().liveGenerationByThreadId, before);
  store.clearLiveGenerationForThread("b");
});

test("the adapter settles the run's own thread when it publishes that pass's context_usage", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /const usageKey = liveThreadKey\(serverCancel\);[\s\S]{0,900}store\.settleLiveGeneration\(usageKey\);/,
  );
});
