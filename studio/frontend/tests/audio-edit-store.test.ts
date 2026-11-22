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

const KEY = "unsloth_audio_edit_v1";
const SAVED = {
  source: { kind: "input", id: "in1", name: "take.wav", durationS: 4.7 },
  transcript: "wasn't a human voice.",
  transcriptFor: "in1",
  edited: "wasn't a robot voice.",
  editedTouched: true,
  mode: "delivery",
  delivery: { speed: 1.5, pitchSteps: 2 },
};
store.set(KEY, JSON.stringify({ state: SAVED, version: 1 }));
store.set(
  "unsloth_audio_clone_v1",
  JSON.stringify({ state: { text: "Clone draft." }, version: 1 }),
);

const { AUDIO_EDIT_STORAGE_KEY, useAudioEditStore } = await import(
  "../src/features/audio/stores/audio-edit-store.ts"
);
const { useAudioCloneStore } = await import(
  "../src/features/audio/stores/audio-clone-store.ts"
);
const state = () => useAudioEditStore.getState();

test("the edit draft comes back after a reload", () => {
  assert.equal(AUDIO_EDIT_STORAGE_KEY, KEY);
  for (const [key, value] of Object.entries(SAVED)) {
    assert.deepEqual(state()[key as keyof typeof SAVED], value, key);
  }
});

test("② follows ① until the user types in it, and Reset puts it back", () => {
  const draft = () => [state().edited, state().transcriptFor];
  state().resetEdited();
  assert.deepEqual(draft(), ["wasn't a human voice.", "in1"]);
  state().setTranscript("It was a fine day.", "in2");
  assert.deepEqual(draft(), ["It was a fine day.", "in2"]);
  state().setEdited("It was a rainy day.");
  state().setTranscript("It was a fine day!");
  assert.deepEqual(draft(), ["It was a rainy day.", "in2"]);
  state().setTranscript("A newer saved transcript.", "in2");
  assert.deepEqual(
    [state().transcript, ...draft()],
    ["A newer saved transcript.", "It was a rainy day.", "in2"],
  );
  state().setDelivery({ pitchSteps: 4 });
  assert.deepEqual(state().delivery, { speed: 1.5, pitchSteps: 4 });
  const { state: saved, version } = JSON.parse(store.get(KEY) ?? "{}");
  assert.deepEqual(
    [version, saved.edited, saved.editedTouched],
    [1, "It was a rainy day.", true],
  );
});

test("editing never touches Clone's draft", () => {
  state().setEdited("Something else.");
  state().setSource(null);
  assert.equal(useAudioCloneStore.getState().text, "Clone draft.");
});
