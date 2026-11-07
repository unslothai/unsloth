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
          if (input === "/api/inference/loaded-models" || input === "/v1/models") return { ok: true, status: 200, json: async () => ({ data: [] }) };
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

test("an initialism before a lowercase word stays in its sentence", () => {
  assert.deepEqual(speechText.splitIntoSentences("The U.S. is large. It has fifty states."), [
    "The U.S. is large.",
    "It has fifty states.",
  ]);
  assert.deepEqual(speechText.splitIntoSentences("See e.g. the docs. Then run it."), [
    "See e.g. the docs.",
    "Then run it.",
  ]);
  assert.deepEqual(speechText.splitIntoSentences("See e.g., the docs. In the U.S., prices rose."), [
    "See e.g., the docs.",
    "In the U.S., prices rose.",
  ]);
  // A capital after one is a real boundary.
  assert.deepEqual(speechText.splitIntoSentences("Born in the U.S. He moved later."), [
    "Born in the U.S.",
    "He moved later.",
  ]);
});

test("a dot with no space after it (.75, .env) stays inside its token", () => {
  assert.deepEqual(speechText.splitIntoSentences("Set the threshold to .75. Then retry."), [
    "Set the threshold to .75.",
    "Then retry.",
  ]);
  assert.deepEqual(speechText.splitIntoSentences("It costs $.50 now. Use -.5 here."), ["It costs $.50 now.", "Use -.5 here."]);
  assert.deepEqual(speechText.splitIntoSentences("Run ./train.py to start. Then wait."), [
    "Run ./train.py to start.",
    "Then wait.",
  ]);
  assert.deepEqual(speechText.splitIntoSentences("10\n.75"), ["10", ".75"]);
  assert.deepEqual(speechText.splitIntoSentences("Use .env for the key. Then restart."), [
    "Use .env for the key.",
    "Then restart.",
  ]);
});

test("a subscriber that cancels on the running notification gets no playback", async () => {
  FakeAudio.live = [];
  (globalThis as { Audio?: unknown }).Audio = FakeAudio;
  const { adapter, answers } = load();
  const utterance = new adapter.StudioSpeechSynthesisAdapter().speak(REPLY) as Utterance & {
    subscribe: (cb: () => void) => () => void;
  };
  utterance.subscribe(() => {
    if (utterance.status.type === "running") utterance.cancel();
  });
  await tick();
  answers.shift()?.(undefined);
  await tick();
  assert.equal(FakeAudio.live.length, 0);
  assert.equal(utterance.status.type, "ended");
});

test("a closing quote ends its sentence without swallowing the next one", () => {
  assert.deepEqual(speechText.splitIntoSentences('The sign read "Stop." Then we left.'), [
    'The sign read "Stop."',
    "Then we left.",
  ]);
  assert.deepEqual(speechText.splitIntoSentences("It ended (finally.) Good."), [
    "It ended (finally.)",
    "Good.",
  ]);
});

test("punctuation left after a sentence is never its own chunk", () => {
  assert.deepEqual(speechText.splitIntoSentences("He said \u201cHello.\u201d"), ["He said \u201cHello.\u201d"]);
  assert.deepEqual(speechText.splitIntoSentences('"Done.".'), ['"Done.".']);
  assert.deepEqual(speechText.splitIntoSentences("She said \u201cGo.\u201d Then we left."), [
    "She said \u201cGo.\u201d",
    "Then we left.",
  ]);
});

test("a dot inside a name like llama.cpp or config.json is not a boundary", () => {
  assert.deepEqual(speechText.splitIntoSentences("It runs in llama.cpp, Ollama and vLLM. Then go."), [
    "It runs in llama.cpp, Ollama and vLLM.",
    "Then go.",
  ]);
  assert.deepEqual(speechText.splitIntoSentences("Open config.json now. Visit unsloth.ai today."), [
    "Open config.json now.",
    "Visit unsloth.ai today.",
  ]);
});

test("an abbreviation that ends a sentence still ends it", () => {
  assert.deepEqual(speechText.splitIntoSentences("I brought apples, oranges, etc. Then I went home."), [
    "I brought apples, oranges, etc.",
    "Then I went home.",
  ]);
  assert.deepEqual(speechText.splitIntoSentences("Fruit, nuts, etc. are fine. Ask Dr. Smith. It is Python vs. Rust."), [
    "Fruit, nuts, etc. are fine.",
    "Ask Dr. Smith.",
    "It is Python vs. Rust.",
  ]);
});

test("an initial in a name stays with the surname", () => {
  assert.deepEqual(speechText.splitIntoSentences("J. Smith wrote it. Then he left."), ["J. Smith wrote it.", "Then he left."]);
  assert.deepEqual(speechText.splitIntoSentences("J. R. R. Tolkien wrote it."), ["J. R. R. Tolkien wrote it."]);
  assert.deepEqual(speechText.splitIntoSentences("Ask John F. Kennedy about it."), ["Ask John F. Kennedy about it."]);
  // After a lowercase word a single letter ends the sentence.
  assert.deepEqual(speechText.splitIntoSentences("Pick option A. Then run it."), ["Pick option A.", "Then run it."]);
});
