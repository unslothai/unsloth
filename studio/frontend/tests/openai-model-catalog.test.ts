// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import {
  audioModelsFromCatalog,
  chatModelsFromCatalog,
  loadedSpeechModelId,
  watchAudioModels,
} from "../src/features/settings/api/openai-model-catalog.ts";

test("non-chat entries never reach the chat example picker", () => {
  const models = chatModelsFromCatalog({
    data: [
      { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M" },
      {
        id: "unsloth/Z-Image-Turbo-GGUF",
        loaded: true,
        quant: "Q4_K_S",
        task: "text-to-image",
      },
      { id: "Lightricks/LTX-2", loaded: false, task: "text-to-video" },
      { id: "tiny", loaded: false, task: "automatic-speech-recognition" },
      {
        id: "unsloth/orpheus-3b-0.1-ft-GGUF",
        loaded: false,
        quant: "Q8_0",
        task: "text-to-speech",
      },
      {
        id: "unsloth/Llama-3.2-1B-Instruct-GGUF",
        loaded: false,
        task: "text-generation",
      },
    ],
  });
  assert.deepEqual(models, [
    { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M" },
    {
      id: "unsloth/Llama-3.2-1B-Instruct-GGUF",
      loaded: false,
      quant: undefined,
    },
  ]);
});

test("older servers without a task field list every entry as before", () => {
  const models = chatModelsFromCatalog({
    data: [{ id: "a", loaded: true }, { id: "b" }],
  });
  assert.deepEqual(models, [
    { id: "a", loaded: true, quant: undefined },
    { id: "b", loaded: false, quant: undefined },
  ]);
});

test("malformed bodies yield no models", () => {
  assert.deepEqual(chatModelsFromCatalog(null), []);
  assert.deepEqual(chatModelsFromCatalog({}), []);
  assert.deepEqual(
    chatModelsFromCatalog({ data: [{ id: 3 }, { id: "" }, {}] }),
    [],
  );
});

test("the audio examples see only speech and transcription models", () => {
  const models = audioModelsFromCatalog({
    data: [
      { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M" },
      { id: "unsloth/Z-Image-Turbo-GGUF", loaded: true, task: "text-to-image" },
      { id: "tiny", loaded: true, task: "automatic-speech-recognition" },
      {
        id: "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF",
        loaded: false,
        task: "text-to-speech",
      },
      { id: "", task: "text-to-speech" },
    ],
  });
  assert.deepEqual(models, [
    { id: "tiny", loaded: true, task: "automatic-speech-recognition" },
    {
      id: "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF",
      loaded: false,
      task: "text-to-speech",
    },
  ]);
  assert.deepEqual(audioModelsFromCatalog(null), []);
});

test("the audio examples keep the workflows a row advertises, separators included", () => {
  const models = audioModelsFromCatalog({
    data: [
      {
        id: "MiniMaxAI/MiniMax-Music3",
        loaded: true,
        task: "text-to-speech",
        audio_workflows: ["music"],
      },
      {
        id: "audio-cpp/audio.cpp-gguf/BS-RoFormer-GGUF",
        loaded: true,
        task: "audio-to-audio",
        audio_workflows: ["separate", 3],
      },
    ],
  });
  assert.deepEqual(models, [
    {
      id: "MiniMaxAI/MiniMax-Music3",
      loaded: true,
      task: "text-to-speech",
      workflows: ["music"],
    },
    {
      id: "audio-cpp/audio.cpp-gguf/BS-RoFormer-GGUF",
      loaded: true,
      task: "audio-to-audio",
      workflows: ["separate"],
    },
  ]);
});

test("the audio catalog is relisted until a model is resident, then more slowly", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const answers = [
      [{ id: "a", task: "text-to-speech" }],
      null,
      [{ id: "a", task: "text-to-speech", loaded: true }],
      [],
    ];
    let calls = 0;
    const seen: unknown[] = [];
    const list = () => {
      const next = answers[calls++];
      return (
        next === null
          ? Promise.reject(new Error("offline"))
          : Promise.resolve(next ?? [])
      ) as Promise<never>;
    };
    const flush = () => new Promise<void>((r) => setImmediate(r));
    const stop = watchAudioModels(list, (models) => seen.push(models));
    await flush();
    assert.equal(calls, 1);
    mock.timers.tick(15000);
    await flush();
    // A failed relist keeps the last answer and retries at the fast pace.
    assert.equal(calls, 2);
    assert.equal(seen.length, 1);
    mock.timers.tick(15000);
    await flush();
    assert.equal(calls, 3);
    assert.equal(seen.length, 2);
    // Something is resident: the next relist waits the slower beat.
    mock.timers.tick(15000);
    await flush();
    assert.equal(calls, 3);
    mock.timers.tick(45000);
    await flush();
    assert.equal(calls, 4);
    stop();
    mock.timers.tick(120000);
    await flush();
    assert.equal(calls, 4);
  } finally {
    mock.timers.reset();
  }
});

test("read aloud picks a loaded model that can speak text", () => {
  const speak = (id: string, loaded: boolean, workflows: string[]) => ({
    id,
    loaded,
    task: "text-to-speech" as const,
    workflows,
  });
  assert.equal(loadedSpeechModelId([]), null);
  // Music, convert-only, transcription and unloaded models cannot read a reply.
  assert.equal(
    loadedSpeechModelId([
      speak("music", true, ["music"]),
      speak("rvc", true, ["convert"]),
      speak("kokoro", false, ["speak"]),
      { id: "whisper", loaded: true, task: "automatic-speech-recognition" },
    ]),
    null,
  );
  const both = [
    speak("kokoro", true, ["speak"]),
    speak("pocket", true, ["speak", "clone"]),
  ];
  assert.equal(loadedSpeechModelId(both), "kokoro");
  // A saved voice needs a model that can clone, when one is loaded.
  assert.equal(loadedSpeechModelId(both, true), "pocket");
  assert.equal(loadedSpeechModelId([both[0]], true), "kokoro");
});

test("a saved voice also counts a loaded clone-only model", () => {
  const cloneOnly = {
    id: "qwen3-tts-base",
    loaded: true,
    task: "text-to-speech" as const,
    workflows: ["clone"],
  };
  // The server runs a saved voice as a clone, so a clone-only model can read it.
  assert.equal(loadedSpeechModelId([cloneOnly], true), "qwen3-tts-base");
  // Without a saved voice it cannot speak plain text.
  assert.equal(loadedSpeechModelId([cloneOnly], false), null);
});
