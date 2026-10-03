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
  loadedModelRunsWorkflow,
  MUSIC_AUDIO_TYPES,
  slotForWorkflow,
  workflowForLoadedModel,
} = await import("../src/features/audio/workflows.ts");

test("the Audio page offers Speak, Clone, Edit, Music and Transcribe in that order", () => {
  assert.deepEqual(
    AUDIO_WORKFLOWS.map((tab) => tab.id),
    ["speak", "clone", "edit", "music", "transcribe"],
  );
  assert.equal(audioWorkflowTab("speak").heading, "Text to speech");
  assert.equal(audioWorkflowTab("clone").heading, "Clone a voice");
  assert.equal(audioWorkflowTab("edit").heading, "Edit speech");
  assert.equal(audioWorkflowTab("edit").label, "Edit");
  assert.equal(audioWorkflowTab("music").heading, "Create music");
  assert.equal(audioWorkflowTab("transcribe").heading, "Transcribe");
});

test("Speak, Clone and Music share the main slot; only Transcribe uses the sidecar", () => {
  assert.equal(slotForWorkflow("speak"), "speak");
  assert.equal(slotForWorkflow("clone"), "speak");
  assert.equal(slotForWorkflow("edit"), "speak");
  assert.equal(slotForWorkflow("music"), "speak");
  assert.equal(slotForWorkflow("transcribe"), "transcribe");
});

test("Create|Train is offered on Speak, Clone and Transcribe, not Music", () => {
  assert.equal(audioWorkflowTab("speak").createTrain, true);
  assert.equal(audioWorkflowTab("clone").createTrain, true);
  assert.equal(audioWorkflowTab("edit").createTrain, false);
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
  assert.equal(clipWorkflow({ workflow: "clone", audio_type: "audiocpp_tts" }), "clone");
  assert.equal(clipWorkflow({ workflow: "edit", audio_type: "audiocpp_tts" }), "edit");
  assert.equal(clipWorkflow({ workflow: "separate", audio_type: "audiocpp_music" }), "music");
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

test("a loaded model keeps the open page when it can run it, else opens its own", () => {
  assert.equal(
    workflowForLoadedModel({ current: "clone", audioWorkflows: ["speak", "clone"], music: false }),
    "clone",
  );
  assert.equal(
    workflowForLoadedModel({ current: "speak", audioWorkflows: ["clone"], music: false }),
    "clone",
  );
  assert.equal(
    workflowForLoadedModel({ current: "speak", audioWorkflows: ["transcribe"], music: true }),
    "music",
  );
  assert.equal(workflowForLoadedModel({ current: "clone", audioWorkflows: null, music: false }), "speak");
  assert.equal(workflowForLoadedModel({ current: "speak", audioWorkflows: [], music: true }), "music");
});

test("only a model that lists Clone can run the Clone page", () => {
  assert.equal(loadedModelRunsWorkflow({ workflow: "clone", audioWorkflows: ["speak"], music: false }), false);
  assert.equal(loadedModelRunsWorkflow({ workflow: "clone", audioWorkflows: ["speak", "clone"], music: false }), true);
  assert.equal(loadedModelRunsWorkflow({ workflow: "speak", audioWorkflows: ["clone"], music: false }), false);
  assert.equal(loadedModelRunsWorkflow({ workflow: "clone", audioWorkflows: undefined, music: false }), false);
  assert.equal(loadedModelRunsWorkflow({ workflow: "music", audioWorkflows: undefined, music: true }), true);
});
