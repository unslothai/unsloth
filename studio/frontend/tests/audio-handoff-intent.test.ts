// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { audioRouteIntent, audioWorkflowForPick, validateAudioSearch } =
  await import("../src/features/audio/route-search.ts");

test("every task Audio runs survives validation, including text-to-audio", () => {
  for (const task of [
    "text-to-speech",
    "automatic-speech-recognition",
    "text-to-audio",
  ]) {
    assert.deepEqual(validateAudioSearch({ task }), { task });
  }
});

test("unknown tasks, workflows and audio types are dropped", () => {
  assert.deepEqual(
    validateAudioSearch({
      task: "text-to-image",
      workflow: "clone",
      audioType: "not-a-type",
      loadId: "   ",
    }),
    {},
  );
  assert.deepEqual(
    validateAudioSearch({ workflow: 3, task: ["text-to-speech"] }),
    {},
  );
});

test("the params the route accepted before pass through unchanged", () => {
  const search = {
    model: "unsloth/orpheus-3b-0.1-ft-GGUF",
    quant: "orpheus-3b-0.1-ft-Q4_K_M.gguf",
    ggufQuant: "Q4_K_M",
    task: "text-to-speech",
    audioType: "audiocpp_tts",
    loadId: "load-1",
    item: "clip-1",
  };
  assert.deepEqual(validateAudioSearch({ ...search, extra: "x" }), search);
  assert.deepEqual(validateAudioSearch({ audioType: "minimax_music3" }), {
    audioType: "minimax_music3",
  });
});

test("a workflow param is kept when it names a workflow", () => {
  for (const workflow of ["speak", "music", "transcribe"]) {
    assert.deepEqual(validateAudioSearch({ workflow }), { workflow });
  }
});

test("the task maps to its workflow", () => {
  assert.equal(audioRouteIntent({ task: "text-to-speech" }), "speak");
  assert.equal(
    audioRouteIntent({ task: "automatic-speech-recognition" }),
    "transcribe",
  );
  assert.equal(audioRouteIntent({ task: "text-to-audio" }), "music");
  assert.equal(audioRouteIntent({}), null);
  assert.equal(audioRouteIntent({ task: "text-to-image" }), null);
});

test("an explicit workflow beats the one the task implies", () => {
  assert.equal(
    audioRouteIntent({ task: "text-to-speech", workflow: "music" }),
    "music",
  );
  assert.equal(
    audioRouteIntent({ task: "text-to-audio", workflow: "bogus" }),
    "music",
  );
});

test("a chat pick opens on the workflow its model runs in", () => {
  assert.equal(
    audioWorkflowForPick({ id: "hexgrad/Kokoro-82M", task: "text-to-speech" }),
    "speak",
  );
  assert.equal(
    audioWorkflowForPick({
      id: "openai/whisper-small",
      task: "automatic-speech-recognition",
    }),
    "transcribe",
  );
  assert.equal(
    audioWorkflowForPick({
      id: "MiniMaxAI/MiniMax-Music3",
      task: "text-to-speech",
      audioType: "minimax_music3",
    }),
    "music",
  );
  assert.equal(
    audioWorkflowForPick({
      id: "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF",
      task: "text-to-speech",
    }),
    "music",
  );
  assert.equal(audioWorkflowForPick({ id: "x/y", task: null }), null);
});

test("the route validates through the shared helper", () => {
  const route = readSrc("app/routes/audio.tsx");
  assert.match(route, /validateSearch: validateAudioSearch,/);
});

test("the chat picker forwards the workflow and still routes no text-to-audio rows", () => {
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(
    pickers,
    /workflow:\s*audioWorkflowForPick\(\{\s*id,\s*task: pickedTask,\s*audioType: meta\.audioType,\s*\}\) \?\? undefined,/,
  );
  const tasks = pickers.match(/export const AUDIO_GEN_TASKS = \[([^\]]*)\]/);
  assert.ok(tasks);
  assert.doesNotMatch(tasks[1], /text-to-audio/);
});

test("a chat picker handoff opens its page before the load, not only after it succeeds", () => {
  const handoff = readSrc("features/audio/hooks/use-audio-handoff.ts");
  // The workflow is part of the dedupe key, so the same model sent for another page is handled again.
  assert.match(
    handoff,
    /const key = `\$\{wanted\}\|[^`]*\|\$\{routeSearch\.workflow \?\? ""\}`;/,
  );
  // Switch, then mark handled, then load: a refused switch leaves the handoff in the URL to retry.
  assert.match(
    handoff,
    /if \(busyRef\.current !== null\) return;[\s\S]*?if \(\s*isAudioWorkflowId\(routedWorkflow\) &&\s*!transitionWorkflow\(routedWorkflow\)\s*\) \{\s*return;\s*\}\s*handledRouteModel\.current = key;\s*handleModelSelect\(wanted,/,
  );
});
