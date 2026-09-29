// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createVoiceSlotQueue } from "../src/features/chat/voice/voice-slot-queue.ts";
import {
  getVoiceMode,
  registerVoiceThreadReset,
  requestVoiceThreadReset,
  setVoiceMode,
} from "../src/features/chat/voice/voice-loop-bridge.ts";

test("teardown waits for an in-flight load before a remounted chat loads again", async () => {
  const enqueue = createVoiceSlotQueue();
  const events: string[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const load = enqueue(async () => { events.push("loading"); await gate; events.push("loaded"); });
  const unload = enqueue(async () => { events.push("unloaded"); });
  const next = enqueue(async () => { events.push("new chat loaded"); });
  await Promise.resolve();
  assert.deepEqual(events, ["loading"]);
  release();
  await Promise.all([load, unload, next]);
  assert.deepEqual(events, ["loading", "loaded", "unloaded", "new chat loaded"]);
});

test("a failed load is reported but does not block cleanup", async () => {
  const enqueue = createVoiceSlotQueue();
  const failure = enqueue(async () => { throw new Error("codec failed"); });
  let unloaded = false;
  const cleanup = enqueue(async () => { unloaded = true; });
  await assert.rejects(failure, /codec failed/);
  await cleanup;
  assert.equal(unloaded, true);
});

test("a queued load rechecks whether the voice selection is still active", async () => {
  const enqueue = createVoiceSlotQueue();
  let active = true;
  let loaded = false;
  const load = enqueue(async () => { if (active) loaded = true; });
  active = false;
  await load;
  assert.equal(loaded, false);
});

test("thread switch stops voice even across the engine unmount gap", () => {
  registerVoiceThreadReset(null);
  setVoiceMode("active");
  requestVoiceThreadReset();
  assert.equal(getVoiceMode(), "off");
});

test("thread cleanup observes off before a delayed callback can rearm", () => {
  setVoiceMode("active");
  let cleaned = false;
  registerVoiceThreadReset(() => { assert.equal(getVoiceMode(), "off"); cleaned = true; });
  requestVoiceThreadReset();
  assert.equal(cleaned, true);
  registerVoiceThreadReset(null);
});
