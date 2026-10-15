// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
registerBundlerResolver();

const KEY = "unsloth_voice_settings";

const { useVoiceSettingsStore } = await import(
  "../src/features/settings/stores/voice-settings-store.ts"
);

async function hydrate(
  saved: { state: Record<string, unknown>; version: number } | null,
) {
  if (saved) store.set(KEY, JSON.stringify(saved));
  else store.delete(KEY);
  await useVoiceSettingsStore.persist.rehydrate();
  return useVoiceSettingsStore.getState();
}

function persisted(): { state: Record<string, unknown>; version: number } {
  return JSON.parse(store.get(KEY) ?? "{}");
}

const ALL_ENGINES_HOST = {
  transformers: { available: true },
  audiocpp: { available: true },
  mtmd: { available: true },
};

test("a user with no saved voice settings gets the llama.cpp default", async () => {
  const state = await hydrate(null);
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
});

test("a v1 save on llama.cpp Qwen3-ASR 0.6B keeps it as the user's", async () => {
  const state = await hydrate({
    state: { sttModel: "qwen3-asr-0.6b" },
    version: 1,
  });
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "user");
  assert.equal(persisted().version, 2);
  assert.equal(persisted().state.sttModelSource, "user");
});

test("a v0 save on Whisper Small reaches llama.cpp Qwen3-ASR 0.6B as the user's", async () => {
  const state = await hydrate({ state: { sttModel: "small" }, version: 0 });
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "user");
});

test("a v1 save without a model gets the default", async () => {
  const state = await hydrate({ state: { ttsRate: 1.5 }, version: 1 });
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
  assert.equal(state.ttsRate, 1.5);
});

test("an explicit llama.cpp pick survives hydration and the host", async () => {
  await hydrate({
    state: { sttModel: "qwen3-asr-0.6b", sttModelSource: "user" },
    version: 2,
  });
  useVoiceSettingsStore.getState().adoptHostSttModel(ALL_ENGINES_HOST);
  const state = useVoiceSettingsStore.getState();
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "user");
});

test("a v1 llama.cpp 1.7B pick is left alone as the user's", async () => {
  const state = await hydrate({
    state: { sttModel: "qwen3-asr-1.7b" },
    version: 1,
  });
  assert.equal(state.sttModel, "qwen3-asr-1.7b");
  assert.equal(state.sttModelSource, "user");
});

test("a v1 Whisper pick settles once: Transformers with PyTorch, llama.cpp without", async () => {
  let state = await hydrate({ state: { sttModel: "large-v3" }, version: 1 });
  assert.equal(state.sttModel, "large-v3");
  assert.equal(state.sttModelSource, "whisper-cpp");
  state.adoptHostSttModel(ALL_ENGINES_HOST);
  state = useVoiceSettingsStore.getState();
  assert.equal(state.sttModel, "large-v3");
  assert.equal(state.sttModelSource, "user");

  state = await hydrate({ state: { sttModel: "small" }, version: 1 });
  state.adoptHostSttModel({
    ...ALL_ENGINES_HOST,
    transformers: { available: false },
  });
  state = useVoiceSettingsStore.getState();
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
  assert.equal(persisted().state.sttModel, "qwen3-asr-0.6b");

  // audio.cpp's twin only when llama.cpp cannot serve it either.
  state = await hydrate({ state: { sttModel: "base" }, version: 1 });
  state.adoptHostSttModel({
    transformers: { available: false },
    audiocpp: { available: true },
    mtmd: { available: false },
  });
  assert.equal(
    useVoiceSettingsStore.getState().sttModel,
    "audiocpp-qwen3-asr-0.6b",
  );
});

test("the untouched default stays put whatever the host has", async () => {
  let state = await hydrate(null);
  state.adoptHostSttModel({ ...ALL_ENGINES_HOST, mtmd: { available: false } });
  state = useVoiceSettingsStore.getState();
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
});

test("picking a model records it as the user's; a language reset records the default", async () => {
  let state = await hydrate(null);
  state.setSttModel("audiocpp-qwen3-asr-0.6b");
  assert.equal(useVoiceSettingsStore.getState().sttModelSource, "user");
  state.setSttModel("audiocpp-moonshine-tiny");
  state = useVoiceSettingsStore.getState();
  assert.equal(state.sttModel, "audiocpp-moonshine-tiny");
  assert.equal(state.sttModelSource, "user");
  assert.equal(persisted().state.sttModelSource, "user");
  // Moonshine is English-only, so Japanese resets it to the default.
  state.setDictationLanguage("ja-JP");
  state = useVoiceSettingsStore.getState();
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
  // an incompatible pick falls back to the default, not the user's.
  state.setSttModel("audiocpp-moonshine-small");
  state = useVoiceSettingsStore.getState();
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
});

test("an invalid or language-incompatible saved model is replaced by the default's", async () => {
  let state = await hydrate({
    state: { sttModel: "not a model", sttModelSource: "user" },
    version: 2,
  });
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
  state = await hydrate({
    state: {
      sttModel: "audiocpp-moonshine-tiny",
      sttModelSource: "user",
      dictationLanguage: "ja-JP",
    },
    version: 2,
  });
  assert.equal(state.sttModel, "qwen3-asr-0.6b");
  assert.equal(state.sttModelSource, "default");
  state = await hydrate({
    state: { sttModel: "small", sttModelSource: "bogus" },
    version: 2,
  });
  assert.equal(state.sttModelSource, "user");
});
