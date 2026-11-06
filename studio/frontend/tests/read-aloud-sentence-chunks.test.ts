// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read aloud on a backend engine speaks a reply sentence by sentence: the first sentence alone
// for a fast start, the next chunk synthesizing while the current one plays.

import assert from "node:assert/strict";
import test from "node:test";

import * as speechText from "../src/features/chat/voice/speech-text.ts";
import * as ttsChunkPlan from "../src/features/chat/voice/tts-chunk-plan.ts";
import * as modelCatalog from "../src/features/settings/api/openai-model-catalog.ts";
import * as formatError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Utterance = { status: { type: string }; cancel: () => void };
type Adapter = {
  readAloudChunks: (text: string) => string[];
  StudioSpeechSynthesisAdapter: new () => { speak: (text: string) => Utterance };
};

class FakeAudio {
  static live: FakeAudio[] = [];
  listeners: Record<string, () => void> = {};
  paused = false;
  playbackRate = 1;
  volume = 1;
  src: string;
  constructor(src: string) {
    this.src = src;
    FakeAudio.live.push(this);
  }
  addEventListener(type: string, fn: () => void) {
    this.listeners[type] = fn;
  }
  removeAttribute() {}
  pause() {
    this.paused = true;
  }
  async play() {}
  end() {
    this.listeners.ended?.();
  }
}

function load() {
  const spoken: string[] = [];
  const answers: ((v: unknown) => void)[] = [];
  const adapter = loadWithStubs<Adapter>(
    new URL("../src/features/chat/adapters/studio-speech-synthesis-adapter.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (input: string, init?: { body?: string }) => {
          if (input === "/v1/models") return { ok: true, status: 200, json: async () => ({ data: [] }) };
          spoken.push(JSON.parse(init?.body ?? "{}").messages[0].content);
          await new Promise((resolve) => answers.push(resolve));
          return { ok: true, status: 200, json: async () => ({ audio: { data: String(spoken.length) } }) };
        },
      },
      "../../settings/api/openai-model-catalog": modelCatalog,
      "@/lib/format-fastapi-error": formatError,
      "../search-images/search-images": { stripSearchImageTokens: (text: string) => text },
      "../voice/speech-text": speechText,
      "../voice/tts-chunk-plan": ttsChunkPlan,
      "../utils/speech-text": { markdownToSpeechText: (text: string) => text },
      "../stores/external-providers-store": { useExternalProvidersStore: { getState: () => ({}) } },
      "../api/providers-api": { encryptProviderApiKey: async () => "" },
      "../external-providers": { getExternalProviderApiKey: () => "" },
      "@/features/settings/stores/voice-settings-store": {
        useVoiceSettingsStore: {
          getState: () => ({ ttsEngine: "studio", ttsStudioVoiceId: "", ttsRate: 1, ttsVolume: 1 }),
        },
      },
      "@/lib/toast": { toast: { error: () => {}, warning: () => {} } },
      "@/features/audio/audio-workspace-constants": { TTS_MAX_TOKENS: 1000 },
    },
  );
  return { adapter, spoken, answers };
}

const tick = async (n = 20) => {
  for (let i = 0; i < n; i++) await new Promise((resolve) => setTimeout(resolve, 0));
};

const REPLY =
  "Paris is the capital of France. It sits on the Seine. It has about two million people. " +
  "The Eiffel Tower was finished in 1889 for the World's Fair. Dr. Smith said it was 3.5 times taller than planned.";

test("the first chunk is one sentence; later ones merge up to the budget without splitting one", () => {
  const { adapter } = load();
  const chunks = adapter.readAloudChunks(REPLY);
  assert.equal(chunks[0], "Paris is the capital of France.");
  assert.ok(chunks.length >= 2);
  for (const chunk of chunks.slice(1)) assert.ok(chunk.length <= ttsChunkPlan.TTS_CHUNK_MAX_CHARS);
  assert.equal(chunks.join(" "), REPLY);
  assert.ok(chunks.some((c) => c.includes("Dr. Smith said it was 3.5 times")));
});

test("a one-sentence reply is one request, as before", () => {
  const { adapter } = load();
  assert.deepEqual(adapter.readAloudChunks("Done."), ["Done."]);
});

test("playback starts after the first chunk and the next one synthesizes while it plays", async () => {
  FakeAudio.live = [];
  (globalThis as { Audio?: unknown }).Audio = FakeAudio;
  const { adapter, spoken, answers } = load();
  const utterance = new adapter.StudioSpeechSynthesisAdapter().speak(REPLY);
  await tick();
  assert.deepEqual(spoken, ["Paris is the capital of France."]);
  answers.shift()?.(undefined);
  await tick();
  assert.equal(utterance.status.type, "running", "the first chunk plays before the rest exists");
  assert.equal(FakeAudio.live.length, 1);
  assert.equal(spoken.length, 2, "the second chunk is already synthesizing during playback");
  const total = adapter.readAloudChunks(REPLY).length;
  for (let i = 1; i < total; i++) {
    answers.shift()?.(undefined);
    await tick();
    assert.notEqual(utterance.status.type, "ended");
    FakeAudio.live[i - 1]?.end();
    await tick();
  }
  FakeAudio.live[total - 1]?.end();
  await tick();
  assert.equal(FakeAudio.live.length, total);
  assert.equal(utterance.status.type, "ended");
  assert.deepEqual(spoken, adapter.readAloudChunks(REPLY));
});

test("cancel stops playback and requests nothing further", async () => {
  FakeAudio.live = [];
  (globalThis as { Audio?: unknown }).Audio = FakeAudio;
  const { adapter, spoken, answers } = load();
  const utterance = new adapter.StudioSpeechSynthesisAdapter().speak(REPLY);
  await tick();
  answers.shift()?.(undefined);
  await tick();
  utterance.cancel();
  answers.shift()?.(undefined);
  await tick();
  assert.equal(FakeAudio.live[0]?.paused, true);
  assert.equal(FakeAudio.live.length, 1, "the chunk that landed after cancel never plays");
  assert.equal(spoken.length, 2);
  assert.equal(utterance.status.type, "ended");
});
