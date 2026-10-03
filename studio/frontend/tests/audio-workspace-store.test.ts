// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const { useAudioWorkspaceStore } = await import(
  "../src/features/audio/stores/audio-workspace-store.ts"
);

test("the workspace opens on Speak with nothing requested", () => {
  const state = useAudioWorkspaceStore.getState();
  assert.equal(state.workflow, "speak");
  assert.equal(state.requestedWorkflow, null);
  assert.equal(state.navExpanded, false);
});

test("a first look at the loaded model opens its page only until something chooses one", () => {
  const store = useAudioWorkspaceStore;
  store.getState().adoptWorkflow("music");
  assert.equal(store.getState().workflow, "music");
  store.getState().commitWorkflow("speak");
  store.getState().adoptWorkflow("music");
  assert.equal(store.getState().workflow, "speak");
});

test("a request does not move the committed workflow until the page takes it", () => {
  const store = useAudioWorkspaceStore;
  store.getState().requestWorkflow("music");
  assert.equal(store.getState().workflow, "speak");
  assert.equal(store.getState().requestedWorkflow, "music");
  store.getState().commitWorkflow("music");
  store.getState().clearRequestedWorkflow();
  assert.equal(store.getState().workflow, "music");
  assert.equal(store.getState().requestedWorkflow, null);
  store.getState().commitWorkflow("speak");
});
