// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
const { AUDIO_WORKSPACE_STORAGE_KEY, useAudioWorkspaceStore } = await import(
  "../src/features/audio/stores/audio-workspace-store.ts"
);

test("the workspace opens on Speak with nothing requested", () => {
  const state = useAudioWorkspaceStore.getState();
  assert.equal(state.workflow, "speak");
  assert.equal(state.requestedWorkflow, null);
  assert.equal(state.navExpanded, false);
  assert.deepEqual(state.lastModelByWorkflow, {});
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

test("only the last model per workflow is persisted", () => {
  const state = useAudioWorkspaceStore.getState();
  state.rememberModel("transcribe", "moonshine/tiny");
  state.rememberModel("speak", "kokoro");
  state.setNavExpanded(true);
  state.commitWorkflow("transcribe");
  const saved = JSON.parse(store.get(AUDIO_WORKSPACE_STORAGE_KEY) ?? "{}");
  assert.equal(AUDIO_WORKSPACE_STORAGE_KEY, "unsloth_audio_workspace");
  assert.equal(saved.version, 1);
  assert.deepEqual(saved.state, {
    lastModelByWorkflow: { transcribe: "moonshine/tiny", speak: "kokoro" },
  });
});

test("remembering the same model again leaves the state untouched", () => {
  const before = useAudioWorkspaceStore.getState().lastModelByWorkflow;
  useAudioWorkspaceStore.getState().rememberModel("speak", "kokoro");
  assert.equal(useAudioWorkspaceStore.getState().lastModelByWorkflow, before);
});

test("any committed switch settles a pending request, so it never fires later", () => {
  const ws = useAudioWorkspaceStore;
  ws.getState().requestWorkflow("music");
  ws.getState().commitWorkflow("transcribe");
  assert.equal(ws.getState().requestedWorkflow, null);
  ws.getState().commitWorkflow("speak");
});
