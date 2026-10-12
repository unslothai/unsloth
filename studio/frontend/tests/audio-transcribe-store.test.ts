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

const saved = {
  source: { kind: "clip", id: "c1", name: "Interview", durationS: 61 },
  ...{ language: "de", timestamps: true, speakers: false, view: "segments" },
};
store.set(
  "unsloth_audio_transcribe_v1",
  JSON.stringify({ state: saved, version: 1 }),
);

const { AUDIO_TRANSCRIBE_STORAGE_KEY, useAudioTranscribeStore } = await import(
  "../src/features/audio/stores/audio-transcribe-store.ts"
);

test("Transcribe's settings come back after a reload and are written back under the versioned key", () => {
  assert.deepEqual(useAudioTranscribeStore.getState(), saved);
  const next = { ...saved, source: null, view: "text" as const };
  useAudioTranscribeStore.setState(next);
  assert.deepEqual(
    JSON.parse(store.get(AUDIO_TRANSCRIBE_STORAGE_KEY) ?? "{}"),
    {
      state: next,
      version: 1,
    },
  );
});
