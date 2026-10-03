// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { audioRowMatchesWorkflow } = await import(
  "../src/features/audio/picker-filter.ts"
);
const { audioModelsForTask, isMusicGenerationModel } = await import(
  "../src/features/audio/catalog.ts"
);

const { audioCppModelSpeaks } = await import(
  "../src/features/audio/audio-cpp-catalog.ts"
);

type Workflow = "speak" | "clone" | "edit" | "music" | "transcribe";
const WORKFLOWS: Workflow[] = ["speak", "clone", "edit", "music", "transcribe"];

function workflowsFor(row: Parameters<typeof audioRowMatchesWorkflow>[0]) {
  return WORKFLOWS.filter((workflow) => audioRowMatchesWorkflow(row, workflow));
}

test("the backend's audio_workflows decides when the row has it", () => {
  assert.deepEqual(
    workflowsFor({
      id: "x/y",
      task: "text-to-speech",
      audioWorkflows: ["music"],
    }),
    ["music"],
  );
});

test("a speech row lists only on Speak", () => {
  assert.deepEqual(
    workflowsFor({ id: "hexgrad/Kokoro-82M", task: "text-to-speech" }),
    ["speak"],
  );
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF",
      task: "text-to-speech",
      audioType: "audiocpp_tts",
    }),
    ["speak"],
  );
});

test("a clone-only speech model lists only on Clone", () => {
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/Chatterbox-GGUF",
      task: "text-to-speech",
      audioWorkflows: ["clone"],
    }),
    ["clone"],
  );
  // Before the backend row says so, the catalog entry does.
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF",
      task: "text-to-speech",
      audioType: "audiocpp_tts",
    }),
    ["clone"],
  );
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/VoxCPM2-GGUF",
      task: "text-to-speech",
    }),
    ["speak", "clone"],
  );
});

test("Speak's catalog rows leave out the clone-only models", () => {
  const speakIds = audioModelsForTask("tts")
    .map((model) => model.id)
    .filter((id) => audioCppModelSpeaks(id));
  assert.ok(speakIds.includes("audio-cpp/audio.cpp-gguf/VoxCPM2-GGUF"));
  assert.ok(speakIds.includes("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF"));
  for (const cloneOnly of [
    "audio-cpp/audio.cpp-gguf/Chatterbox-GGUF",
    "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF",
    "audio-cpp/audio.cpp-gguf/IndexTTS2-GGUF",
    "audio-cpp/audio.cpp-gguf/CosyVoice3-GGUF",
  ]) {
    assert.ok(!speakIds.includes(cloneOnly), cloneOnly);
  }
});

test("a music GGUF (text-to-audio, no audio type) lists only on Music", () => {
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/Stable-Audio-3-Small-Music-GGUF",
      task: "text-to-audio",
      audioType: null,
    }),
    ["music"],
  );
});

test("MiniMax Music 3 is tagged text-to-speech but lists only on Music", () => {
  assert.deepEqual(
    workflowsFor({
      id: "MiniMaxAI/MiniMax-Music3",
      task: "text-to-speech",
      audioType: "minimax_music3",
    }),
    ["music"],
  );
});

test("an ASR row lists only on Transcribe", () => {
  assert.deepEqual(
    workflowsFor({
      id: "openai/whisper-small",
      task: "automatic-speech-recognition",
    }),
    ["transcribe"],
  );
});

test("a row with nothing to classify it stays listed", () => {
  assert.deepEqual(workflowsFor({ id: "someone/unknown" }), WORKFLOWS);
});

test("splitting the speech catalog on music leaves no model out and none twice", () => {
  const tts = audioModelsForTask("tts");
  const speak = tts.filter((model) => !isMusicGenerationModel(model.id));
  const music = tts.filter((model) => isMusicGenerationModel(model.id));
  assert.ok(speak.length > 0);
  assert.ok(music.length > 0);
  const speakIds = new Set(speak.map((model) => model.id));
  assert.equal(music.filter((model) => speakIds.has(model.id)).length, 0);
  assert.deepEqual(
    [...speak, ...music].map((model) => model.id).sort(),
    tts.map((model) => model.id).sort(),
  );
});

test("an undefined rowFilter leaves the picker rows untouched", () => {
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  const guards = pickers.match(/\(!rowFilter \|\|\s*rowFilter\(\{/g) ?? [];
  // Cached GGUF and cached repos only; Hub search rows stay unfiltered.
  assert.equal(guards.length, 2);
  const selector = readSrc(
    "features/model-picker/components/model-selector.tsx",
  );
  assert.match(selector, /rowFilter=\{rowFilter\}/);
});

test("a speech editing model lists on Edit beside its other pages", () => {
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/DotTTS-Edit-GGUF",
      task: "text-to-speech",
      audioWorkflows: ["speak", "edit"],
    }),
    ["speak", "edit"],
  );
  // DotTTS's other packages do not edit.
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/DotTTS-MF-GGUF",
      task: "text-to-speech",
      audioWorkflows: ["speak"],
    }),
    ["speak"],
  );
  // Without the backend's list, the catalog seed decides.
  assert.deepEqual(
    workflowsFor({
      id: "audio-cpp/audio.cpp-gguf/Vevo2-GGUF",
      task: "text-to-speech",
      audioType: "audiocpp_tts",
    }),
    ["clone", "edit"],
  );
});
