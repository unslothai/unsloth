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
    source: { kind: "input", id: "in1", name: "take.wav", durationS: 4.7 },
    transcript: "wasn't a human voice.",
    transcriptFor: "in1",
    edited: "wasn't a robot voice.",
    editedTouched: true,
    mode: "delivery",
    delivery: { speed: 1.5, pitchSteps: 2 },
  },
  version: 1,
};
store.set("unsloth_audio_edit_v1", JSON.stringify(SAVED));
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

test("the edit draft comes back after a reload", () => {
  assert.equal(AUDIO_EDIT_STORAGE_KEY, "unsloth_audio_edit_v1");
  const state = useAudioEditStore.getState();
  assert.deepEqual(state.source, SAVED.state.source);
  assert.equal(state.transcript, "wasn't a human voice.");
  assert.equal(state.transcriptFor, "in1");
  assert.equal(state.edited, "wasn't a robot voice.");
  assert.equal(state.mode, "delivery");
  assert.deepEqual(state.delivery, { speed: 1.5, pitchSteps: 2 });
});

test("② follows ① until the user types in it, and Reset puts it back", () => {
  const edit = useAudioEditStore.getState();
  edit.resetEdited();
  assert.equal(useAudioEditStore.getState().edited, "wasn't a human voice.");
  edit.setTranscript("It was a fine day.", "in2");
  assert.equal(useAudioEditStore.getState().edited, "It was a fine day.");
  assert.equal(useAudioEditStore.getState().transcriptFor, "in2");
  edit.setEdited("It was a rainy day.");
  edit.setTranscript("It was a fine day!");
  const state = useAudioEditStore.getState();
  assert.equal(state.edited, "It was a rainy day.");
  // Leaving transcriptFor out keeps the source it belongs to.
  assert.equal(state.transcriptFor, "in2");
  edit.setDelivery({ pitchSteps: 4 });
  assert.deepEqual(useAudioEditStore.getState().delivery, {
    speed: 1.5,
    pitchSteps: 4,
  });
  const persisted = JSON.parse(store.get("unsloth_audio_edit_v1") ?? "{}");
  assert.equal(persisted.state.edited, "It was a rainy day.");
  assert.equal(persisted.state.editedTouched, true);
});

test("editing never touches Clone's draft", () => {
  useAudioEditStore.getState().setEdited("Something else.");
  useAudioEditStore.getState().setSource(null);
  assert.equal(useAudioCloneStore.getState().text, "Clone draft.");
});
