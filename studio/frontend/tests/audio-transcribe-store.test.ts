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
    source: { kind: "clip", id: "c1", name: "Interview", durationS: 61 },
    language: "de",
    timestamps: true,
    speakers: false,
    view: "segments",
  },
  version: 1,
};
store.set("unsloth_audio_transcribe_v1", JSON.stringify(SAVED));

const { AUDIO_TRANSCRIBE_STORAGE_KEY, useAudioTranscribeStore } = await import(
  "../src/features/audio/stores/audio-transcribe-store.ts"
);

test("Transcribe's draft comes back after a reload", () => {
  assert.equal(AUDIO_TRANSCRIBE_STORAGE_KEY, "unsloth_audio_transcribe_v1");
  const state = useAudioTranscribeStore.getState();
  assert.deepEqual(state.source, SAVED.state.source);
  assert.equal(state.language, "de");
  assert.equal(state.timestamps, true);
  assert.equal(state.speakers, false);
  assert.equal(state.view, "segments");
});

test("changes are written back under the versioned key, setters only", () => {
  const state = useAudioTranscribeStore.getState();
  state.setSource(null);
  state.setLanguage("");
  state.setTimestamps(false);
  state.setSpeakers(true);
  state.setView("text");
  const written = JSON.parse(store.get("unsloth_audio_transcribe_v1") ?? "{}");
  assert.equal(written.version, 1);
  assert.deepEqual(written.state, {
    source: null,
    language: "",
    timestamps: false,
    speakers: true,
    view: "text",
  });
});
