// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";
import { inFlightThreadReads } from "../src/features/chat/api/thread-read-cache.ts";

const { store: storage, fireWindowEvent } = installLocalStorageFake();
register("./thread-sampling-resolver.mjs", import.meta.url);
const runtime = await import("../src/features/chat/stores/chat-runtime-store.ts");

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
const pending = () => ({
  promise: new Promise<null>(() => {}),
  controller: new AbortController(),
  readers: 1,
  startedAt: performance.now(),
  authEpoch: 0,
});

test("settings replay prevents later readers from joining the pre-replay request", async () => {
  const before = pending();
  inFlightThreadReads.set("replay", before);
  const unrelated = pending();
  inFlightThreadReads.set("other", unrelated);
  storage.set("unsloth_chat_thread_settings_replay", JSON.stringify({
    replay: { settings: { temperature: 0.9 } },
  }));
  runtime.replayUnconfirmedThreadSettings();
  await tick();
  assert.equal(inFlightThreadReads.has("replay"), false);
  assert.equal(inFlightThreadReads.get("other"), unrelated);
  assert.equal(before.controller.signal.aborted, false);
  inFlightThreadReads.clear();
});

test("a terminal settings write also invalidates reads without aborting existing readers", async () => {
  storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
  const store = runtime.useChatRuntimeStore;
  await store.getState().hydratePersistedSettings();
  store.getState().setActiveThreadId("beacon");
  runtime.beginThreadScopedPairing("beacon");
  store.getState().setParams({ ...store.getState().params, temperature: 0.7 });
  const before = pending();
  inFlightThreadReads.set("beacon", before);
  assert.ok(fireWindowEvent("pagehide", {}) > 0);
  await tick();
  assert.equal(inFlightThreadReads.has("beacon"), false);
  assert.equal(before.controller.signal.aborted, false);
  await runtime.flushPendingChatSettings();
  await runtime.awaitStartedThreadScopedSettingsWrites();
  inFlightThreadReads.clear();
});
