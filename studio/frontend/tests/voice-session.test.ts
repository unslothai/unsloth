// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { VOICE_CHAT_MODEL_VOICE, VOICE_DEFAULT_TTS_MODEL } from "../src/features/chat/voice/voice-models.ts";
import {
  createVoiceSession,
  type VoiceMode,
  type VoiceSessionPorts,
} from "../src/features/chat/voice/voice-session.ts";

function fakePorts(init: { mode?: VoiceMode; voice?: string; checkpoint?: string; serving?: string[] } = {}) {
  const state = {
    voiceMode: init.mode ?? ("active" as VoiceMode),
    selectedVoiceModelId: init.voice ?? "voice-a",
    checkpoint: init.checkpoint ?? "chat-model",
    modelLoading: false,
  };
  const serving = new Set(init.serving ?? []);
  const log: string[] = [];
  let slotLoaded = false;
  let finishLoad: (() => void) | null = null;
  let finishUnload: (() => void) | null = null;
  let holdUnloads = false;
  const ports: VoiceSessionPorts = {
    state: () => ({ ...state }),
    setSelectedVoiceModelId: (id) => {
      state.selectedVoiceModelId = id;
    },
    setVoiceSlotLoaded: (loaded) => {
      slotLoaded = loaded;
    },
    setVoiceSlotLoading: () => {},
    slot: async () => ({
      voiceServing: async (id) => serving.has(id),
      servedVoice: async () => null,
      loadVoice: (id) =>
        new Promise<void>((resolve) => {
          log.push(`load ${id}`);
          finishLoad = () => {
            serving.add(id);
            resolve();
          };
        }),
      unloadVoice: (id) =>
        new Promise<void>((resolve) => {
          log.push(`unload ${id}`);
          const done = () => {
            serving.delete(id);
            resolve();
          };
          if (holdUnloads) finishUnload = done;
          else done();
        }),
    }),
    releaseListening: () => log.push("releaseListening"),
    chatModelCannotSpeak: () => log.push("cannotSpeak"),
    loadFailed: () => log.push("loadFailed"),
  };
  return {
    ports,
    state,
    log,
    serving,
    slotLoaded: () => slotLoaded,
    finish: () => finishLoad?.(),
    holdUnloads: () => {
      holdUnloads = true;
    },
    finishUnload: () => finishUnload?.(),
  };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test("voice mode going off releases the listening models; other moves leave them", () => {
  const f = fakePorts();
  const session = createVoiceSession(f.ports, "off");
  session.setMode("off");
  session.setMode("configuring");
  session.setMode("active");
  assert.deepEqual(f.log, []);
  session.setMode("off");
  assert.deepEqual(f.log, ["releaseListening"]);
});

test("leaving Chat releases the listening models only while voice mode is on", async () => {
  const off = fakePorts();
  createVoiceSession(off.ports, "off").dispose();
  await tick();
  assert.deepEqual(off.log, []);
  const on = fakePorts();
  createVoiceSession(on.ports, "active").dispose();
  await tick();
  assert.deepEqual(on.log, ["releaseListening"]);
});

test("StrictMode's dev remount (dispose, then activate at once) releases nothing", async () => {
  const f = fakePorts({ serving: ["voice-a"] });
  const session = createVoiceSession(f.ports, "active");
  session.activate();
  assert.equal(await session.ensureVoiceSlotLoaded(), true);
  session.dispose();
  session.activate();
  await tick();
  assert.deepEqual(f.log, []);
  assert.equal(f.slotLoaded(), true);
});

test("a caller that asks after the pick changed waits for the new pick, not the old load", async () => {
  const f = fakePorts();
  const session = createVoiceSession(f.ports, "active");
  const first = session.ensureVoiceSlotLoaded();
  await tick();
  f.state.selectedVoiceModelId = "voice-b";
  const second = session.ensureVoiceSlotLoaded();
  assert.notEqual(second, first);
  f.finish();
  assert.equal(await first, true);
  await tick();
  assert.deepEqual(f.log, ["load voice-a", "unload voice-a", "load voice-b"]);
  f.finish();
  assert.equal(await second, true);
  assert.equal(f.slotLoaded(), true);
});

test("voice off and on again before an old load settles: the new ask loads afresh, not the stale false", async () => {
  const f = fakePorts();
  const session = createVoiceSession(f.ports, "active");
  const stale = session.ensureVoiceSlotLoaded();
  await tick();
  session.setMode("off");
  session.setMode("active");
  const fresh = session.ensureVoiceSlotLoaded();
  f.finish();
  assert.equal(await stale, false);
  await tick();
  f.finish();
  assert.equal(await fresh, true);
  assert.deepEqual(f.log, ["load voice-a", "releaseListening", "unload voice-a", "load voice-a"]);
});

test("voice off then on: a load waits for the release's unload instead of seeing the voice still serving", async () => {
  const f = fakePorts();
  const session = createVoiceSession(f.ports, "active");
  const first = session.ensureVoiceSlotLoaded();
  await tick();
  f.finish();
  assert.equal(await first, true);
  f.holdUnloads();
  session.setMode("off");
  session.setMode("active");
  const again = session.ensureVoiceSlotLoaded();
  await tick();
  assert.equal(f.slotLoaded(), false, "not marked loaded while its unload is still running");
  f.finishUnload();
  await tick();
  f.finish();
  assert.equal(await again, true);
  assert.deepEqual(f.log, ["load voice-a", "releaseListening", "unload voice-a", "load voice-a"]);
  assert.equal(f.slotLoaded(), true);
});

test("voice off while a borrow waits on releasing the old voice leaves the slot unloaded", async () => {
  const f = fakePorts({ serving: ["voice-b"] });
  const session = createVoiceSession(f.ports, "active");
  const own = session.ensureVoiceSlotLoaded();
  await tick();
  f.finish();
  assert.equal(await own, true);
  f.state.selectedVoiceModelId = "voice-b";
  f.holdUnloads();
  const borrow = session.ensureVoiceSlotLoaded();
  await tick();
  session.setMode("off");
  f.finishUnload();
  assert.equal(await borrow, false);
  assert.equal(f.slotLoaded(), false);
});

test("a voice loaded here is unloaded when voice mode goes off", async () => {
  const f = fakePorts();
  const session = createVoiceSession(f.ports, "active");
  const first = session.ensureVoiceSlotLoaded();
  assert.equal(session.ensureVoiceSlotLoaded(), first, "a second ask joins the load in flight");
  await tick();
  f.finish();
  assert.equal(await first, true);
  assert.equal(f.slotLoaded(), true);
  session.setMode("off");
  await tick();
  assert.deepEqual(f.log, ["load voice-a", "releaseListening", "unload voice-a"]);
  assert.equal(f.slotLoaded(), false);
});

test("a load that settles after voice mode let go is unloaded, not kept", async () => {
  const f = fakePorts();
  const session = createVoiceSession(f.ports, "active");
  const load = session.ensureVoiceSlotLoaded();
  await tick();
  session.setMode("off");
  f.finish();
  assert.equal(await load, false);
  assert.deepEqual(f.log, ["load voice-a", "releaseListening", "unload voice-a"]);
});

test("a voice the Audio page already serves is borrowed, never unloaded", async () => {
  const f = fakePorts({ serving: ["voice-a"] });
  const session = createVoiceSession(f.ports, "active");
  assert.equal(await session.ensureVoiceSlotLoaded(), true);
  session.setMode("off");
  await tick();
  assert.deepEqual(f.log, ["releaseListening"]);
});

test("nothing loads outside voice mode or without a chat model", async () => {
  const off = fakePorts({ mode: "off" });
  assert.equal(await createVoiceSession(off.ports, "off").ensureVoiceSlotLoaded(), false);
  const empty = fakePorts({ checkpoint: "" });
  assert.equal(await createVoiceSession(empty.ports, "active").ensureVoiceSlotLoaded(), false);
  assert.deepEqual([...off.log, ...empty.log], []);
});

test("a chat-model voice that cannot speak hands over to the default voice", async () => {
  const f = fakePorts({ voice: VOICE_CHAT_MODEL_VOICE });
  const session = createVoiceSession(f.ports, "active");
  const load = session.ensureVoiceSlotLoaded();
  await tick();
  f.finish();
  assert.equal(await load, true);
  assert.equal(f.state.selectedVoiceModelId, VOICE_DEFAULT_TTS_MODEL);
  assert.deepEqual(f.log, ["cannotSpeak", `load ${VOICE_DEFAULT_TTS_MODEL}`]);
});
