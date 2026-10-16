// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Turning Voice off (or leaving Chat) while the voice model is still loading used
// to find nothing to unload, and the load then settled as the page's own, leaving
// the extra model resident with voice off.

import assert from "node:assert/strict";
import test from "node:test";

import { createVoiceSlotOwner } from "../src/features/chat/voice/voice-slot-owner.ts";

function harness() {
  const calls: string[] = [];
  let finishLoad: () => void = () => {};
  const state = { voiceOff: false, loaded: false };
  const owner = createVoiceSlotOwner({
    api: async () => ({
      voiceServing: async () => false,
      loadVoice: (id) => {
        calls.push(`load ${id}`);
        return new Promise<void>((resolve) => {
          finishLoad = resolve;
        });
      },
      unloadVoice: async (id) => {
        calls.push(`unload ${id}`);
      },
    }),
    pickedVoice: () => "kokoro",
    hasChatModel: () => true,
    voiceOff: () => state.voiceOff,
    setLoaded: (loaded) => {
      state.loaded = loaded;
    },
    setLoading: () => {},
    onError: (error) => {
      throw error;
    },
  });
  const loadStarted = async () => {
    while (!calls.includes("load kokoro"))
      await new Promise((r) => setTimeout(r, 0));
  };
  return { owner, calls, state, loadStarted, finish: () => finishLoad() };
}

test("a load that settles with voice active is kept", async () => {
  const h = harness();
  const run = h.owner.ensure();
  await h.loadStarted();
  h.finish();
  assert.equal(await run, true);
  assert.equal(h.owner.loadedId, "kokoro");
  assert.equal(h.state.loaded, true);
  assert.deepEqual(h.calls, ["load kokoro"]);
});

test("a load that settles after Voice was turned off is unloaded", async () => {
  const h = harness();
  const run = h.owner.ensure();
  await h.loadStarted();
  h.state.voiceOff = true;
  h.owner.unloadOwn();
  h.finish();
  assert.equal(await run, false);
  assert.equal(h.owner.loadedId, null);
  assert.equal(h.state.loaded, false);
  assert.deepEqual(h.calls, ["load kokoro", "unload kokoro"]);
});

test("a load that settles after the chat page closed is unloaded", async () => {
  const h = harness();
  h.owner.open();
  const run = h.owner.ensure();
  await h.loadStarted();
  h.owner.close();
  h.finish();
  assert.equal(await run, false);
  assert.equal(h.owner.loadedId, null);
  assert.deepEqual(h.calls, ["load kokoro", "unload kokoro"]);
});

test("reopening after a close loads again (StrictMode remount)", async () => {
  const h = harness();
  h.owner.close();
  h.owner.open();
  const run = h.owner.ensure();
  await h.loadStarted();
  h.finish();
  assert.equal(await run, true);
  assert.equal(h.owner.loadedId, "kokoro");
});

// A page that left mid-load unloads by model id. A new page adopting the same
// voice in that window must not have it unloaded under it.
test("a new page keeps the voice an old page's stale cleanup would unload", async () => {
  const served = new Set<string>();
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const api = async () => ({
    voiceServing: async (id: string) => served.has(id),
    loadVoice: async (id: string) => {
      await sleep(10);
      served.add(id);
    },
    unloadVoice: async (id: string) => {
      await sleep(5);
      served.delete(id);
    },
  });
  const owner = () =>
    createVoiceSlotOwner({
      api,
      pickedVoice: () => "kokoro",
      hasChatModel: () => true,
      voiceOff: () => false,
      setLoaded: () => {},
      setLoading: () => {},
      onError: (error) => {
        throw error;
      },
    });
  const oldPage = owner();
  oldPage.open();
  const oldRun = oldPage.ensure();
  await sleep(1);
  oldPage.close();
  const newPage = owner();
  newPage.open();
  assert.equal(await newPage.ensure(), true);
  assert.equal(await oldRun, false);
  await sleep(30);
  assert.equal(newPage.loadedId, "kokoro");
  assert.ok(served.has("kokoro"), "the new page's voice was unloaded under it");
});
