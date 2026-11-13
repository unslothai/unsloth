// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { useSettingsDialogStore } from "../src/features/settings/stores/settings-dialog-store.ts";

const store = useSettingsDialogStore;
const REQUEST = {
  workflow: "separate",
  model: "audio-cpp/audio.cpp-gguf/HTDemucs-GGUF",
} as const;

function reset() {
  store.setState({
    open: false,
    activeTab: "general",
    scrollTarget: null,
    audioApiRequested: null,
  });
}

test("Use via API opens API keys at the Audio API card", () => {
  reset();
  store.getState().openAudioApi(REQUEST);
  const s = store.getState();
  assert.equal(s.open, true);
  assert.equal(s.activeTab, "api-keys");
  assert.equal(s.scrollTarget, "api-keys-audio-api");
  assert.deepEqual(s.audioApiRequested, REQUEST);
  s.consumeAudioApiRequest();
  assert.equal(store.getState().audioApiRequested, null);
});

test("an unconsumed request does not outlive leaving the tab or closing", () => {
  reset();
  store.getState().openAudioApi(REQUEST);
  // Reselecting the tab keeps it while the card's chunk may still be loading.
  store.getState().setActiveTab("api-keys");
  assert.deepEqual(store.getState().audioApiRequested, REQUEST);
  store.getState().setActiveTab("general");
  assert.equal(store.getState().audioApiRequested, null);

  store.getState().openAudioApi(REQUEST);
  store.getState().closeDialog();
  assert.equal(store.getState().audioApiRequested, null);

  store.getState().openAudioApi(REQUEST);
  store.getState().openConnectionSettings("provider-1");
  assert.equal(store.getState().audioApiRequested, null);
});
