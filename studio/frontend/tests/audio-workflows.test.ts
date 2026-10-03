// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  AUDIO_WORKFLOWS,
  audioWorkflowForAudioType,
  audioWorkflowForTask,
  audioWorkflowTab,
  clipWorkflow,
  isAudioWorkflowId,
  MUSIC_AUDIO_TYPES,
  slotForWorkflow,
} = await import("../src/features/audio/workflows.ts");

test("the Audio page offers Speak, Music and Transcribe in that order", () => {
  assert.deepEqual(
    AUDIO_WORKFLOWS.map((tab) => tab.id),
    ["speak", "music", "transcribe"],
  );
  assert.equal(audioWorkflowTab("speak").heading, "Text to speech");
  assert.equal(audioWorkflowTab("music").heading, "Create music");
  assert.equal(audioWorkflowTab("transcribe").heading, "Transcribe");
});

test("Speak and Music share the main slot; only Transcribe uses the sidecar", () => {
  assert.equal(slotForWorkflow("speak"), "speak");
  assert.equal(slotForWorkflow("music"), "speak");
  assert.equal(slotForWorkflow("transcribe"), "transcribe");
});

test("Create|Train is offered on Speak and Transcribe, not Music", () => {
  assert.equal(audioWorkflowTab("speak").createTrain, true);
  assert.equal(audioWorkflowTab("music").createTrain, false);
  assert.equal(audioWorkflowTab("transcribe").createTrain, true);
});

test("pipeline tags map onto workflows and anything else is ignored", () => {
  assert.equal(audioWorkflowForTask("text-to-speech"), "speak");
  assert.equal(audioWorkflowForTask("automatic-speech-recognition"), "transcribe");
  assert.equal(audioWorkflowForTask("text-to-audio"), "music");
  assert.equal(audioWorkflowForTask("text-generation"), null);
  assert.equal(audioWorkflowForTask(undefined), null);
  assert.equal(isAudioWorkflowId("music"), true);
  assert.equal(isAudioWorkflowId("separate"), false);
});

test("a clip without a workflow falls back to its audio type", () => {
  assert.equal(clipWorkflow({ audio_type: "minimax_music3" }), "music");
  assert.equal(clipWorkflow({ audio_type: "audiocpp_music" }), "music");
  assert.equal(clipWorkflow({ audio_type: "audiocpp_tts" }), "speak");
  assert.equal(clipWorkflow({ audio_type: "snac" }), "speak");
  assert.equal(clipWorkflow({ workflow: "music", audio_type: "snac" }), "music");
  assert.equal(clipWorkflow({ workflow: "edit", audio_type: "audiocpp_music" }), "music");
  assert.equal(audioWorkflowForAudioType(null), "speak");
});

test("the music audio types match the backend's list", () => {
  const backend = readText("../../backend/core/inference/audio_workflows.py");
  const block = backend.match(/MUSIC_AUDIO_TYPES[^=]*=\s*frozenset\(\s*[({]([^)}]*)[)}]/);
  assert.ok(block, "MUSIC_AUDIO_TYPES literal not found in audio_workflows.py");
  const names = new Set(
    [...block[1].matchAll(/"([^"]+)"|([A-Z_]+_AUDIO_TYPE)/g)].map(
      (m) => m[1] ?? (m[2] === "AUDIO_CPP_MUSIC_AUDIO_TYPE" ? "audiocpp_music" : m[2]),
    ),
  );
  assert.deepEqual(names, new Set(MUSIC_AUDIO_TYPES));
});
