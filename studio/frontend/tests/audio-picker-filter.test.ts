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

type Workflow = "speak" | "music" | "transcribe";
const WORKFLOWS: Workflow[] = ["speak", "music", "transcribe"];

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
  // Cached GGUF, cached repos, LM Studio, ./models and custom folders; Hub search rows stay unfiltered.
  assert.equal(guards.length, 5);
  // Each On Device list applies it, so a custom-folder speech model is not offered on the Music page.
  for (const list of [
    "sortedCachedGguf",
    "sortedCachedModels",
    "sortedLmStudio",
    "sortedLocalDir",
    "sortedCustomFolderModels",
  ]) {
    const start = pickers.indexOf(`const ${list} = useMemo(`);
    assert.ok(start >= 0, list);
    const body = pickers.slice(start, pickers.indexOf("\n  );\n", start));
    assert.match(body, /\(!rowFilter \|\|\s*rowFilter\(\{/, list);
    assert.match(body, /rowFilter,\n\s*\]/, `${list} deps`);
  }
  const selector = readSrc(
    "features/model-picker/components/model-selector.tsx",
  );
  assert.match(selector, /rowFilter=\{rowFilter\}/);
});

test("typed Hub search results are scoped to the page too", () => {
  const pickers = readSrc("features/model-picker/components/model-selector/pickers.tsx");
  const body = pickers.slice(pickers.indexOf("const searchIdsFrom = useCallback("));
  const block = body.slice(0, body.indexOf("\n    ],") + 1);
  assert.match(block, /\.filter\(\(r\) => !rowFilter \|\| rowFilter\(\{ id: r\.id, task: r\.pipelineTag \}\)\)/);
  assert.match(block, /\n\s+rowFilter,\n/);
});

test("a Hub music model tagged text-to-speech stays on Music", () => {
  const row = { id: "MiniMaxAI/MiniMax-Music3", task: "text-to-speech" };
  assert.equal(audioRowMatchesWorkflow(row, "speak"), false);
  assert.equal(audioRowMatchesWorkflow(row, "music"), true);
  assert.equal(audioRowMatchesWorkflow({ id: "hexgrad/Kokoro-82M", task: "text-to-speech" }, "speak"), true);
});
