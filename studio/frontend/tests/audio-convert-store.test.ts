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

const SAVED = {
  state: {
    source: { kind: "clip", id: "c1", name: "Take 1", durationS: 8 },
    target: { kind: "voice", id: "v1", name: "Narrator", durationS: 6 },
    builtinVoice: "manthos",
    mode: "singing",
    pitchAuto: false,
    pitch: 3,
    sourceText: "La la la.",
    compareSide: "source",
  },
  version: 1,
};
store.set("unsloth_audio_convert_v1", JSON.stringify(SAVED));

const { AUDIO_CONVERT_STORAGE_KEY, useAudioConvertStore } = await import(
  "../src/features/audio/stores/audio-convert-store.ts"
);

test("the draft comes back after a reload", () => {
  assert.equal(AUDIO_CONVERT_STORAGE_KEY, "unsloth_audio_convert_v1");
  const state = useAudioConvertStore.getState();
  for (const [key, value] of Object.entries(SAVED.state)) {
    assert.deepEqual(state[key as keyof typeof SAVED.state], value, key);
  }
});

test("every draft field is written through, and no action is", () => {
  const state = useAudioConvertStore.getState();
  state.setSource({
    kind: "input",
    id: "abc",
    name: "take.webm",
    durationS: 4,
    expiresAt: "2026-10-04T00:00:00Z",
  });
  state.setTarget(null);
  state.setBuiltinVoice("fraise");
  state.setMode("speech");
  state.setPitchAuto(true);
  state.setPitch(-5);
  state.setSourceText("Hello.");
  state.setCompareSide("converted");
  const saved = JSON.parse(store.get(AUDIO_CONVERT_STORAGE_KEY) ?? "{}");
  assert.equal(saved.version, 1);
  assert.deepEqual(saved.state, {
    source: {
      kind: "input",
      id: "abc",
      name: "take.webm",
      durationS: 4,
      expiresAt: "2026-10-04T00:00:00Z",
    },
    target: null,
    builtinVoice: "fraise",
    mode: "speech",
    pitchAuto: true,
    pitch: -5,
    sourceText: "Hello.",
    compareSide: "converted",
  });
});

test("pitch stays a whole number of semitones within two octaves", () => {
  const state = useAudioConvertStore.getState();
  state.setPitch(30);
  assert.equal(useAudioConvertStore.getState().pitch, 24);
  state.setPitch(-2.6);
  assert.equal(useAudioConvertStore.getState().pitch, -3);
  state.setPitch(Number.NaN);
  assert.equal(useAudioConvertStore.getState().pitch, 0);
});

test("a new recording drops the old transcript or brings its own", () => {
  const s = useAudioConvertStore.getState();
  s.setSource({ kind: "clip", id: "a", name: "A", durationS: 2 });
  s.setSourceText("words of A");
  s.setSource({ kind: "clip", id: "a", name: "A again", durationS: 2 });
  assert.equal(useAudioConvertStore.getState().sourceText, "words of A");
  s.setSource({ kind: "input", id: "b", name: "B", durationS: 3 });
  assert.equal(useAudioConvertStore.getState().sourceText, "");
  s.setSource({
    kind: "voice",
    id: "v",
    name: "V",
    durationS: 4,
    transcript: " words of V ",
  });
  assert.equal(useAudioConvertStore.getState().sourceText, "words of V");
  s.setSource(null);
  assert.equal(useAudioConvertStore.getState().sourceText, "");
});
