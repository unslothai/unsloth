import assert from "node:assert/strict";
import test from "node:test";

import { getImageInputUnavailableReason } from "../src/features/chat/utils/image-input-support.ts";
import { mergeQueuedModelCapabilities } from "../src/features/chat/utils/queued-model-capabilities.ts";

test("status capabilities synthesize an audio model missing from the catalog", () => {
  assert.deepEqual(
    mergeQueuedModelCapabilities([], "local/audio-model", {
      isVision: false,
      isGguf: false,
      isAudio: true,
      audioType: "tts",
      hasAudioInput: false,
    }),
    [
      {
        id: "local/audio-model",
        name: "local/audio-model",
        isLora: false,
        isVision: false,
        isGguf: false,
        isAudio: true,
        audioType: "tts",
        hasAudioInput: false,
      },
    ],
  );
});

test("status capabilities override stale catalog flags", () => {
  assert.deepEqual(
    mergeQueuedModelCapabilities(
      [
        {
          id: "local/audio-model",
          name: "Friendly audio model",
          isVision: true,
          isLora: true,
          isGguf: true,
          isAudio: false,
          audioType: null,
          hasAudioInput: true,
        },
      ],
      "local/audio-model",
      {
        isVision: false,
        isGguf: false,
        isAudio: true,
        audioType: "tts",
        hasAudioInput: false,
      },
    ),
    [
      {
        id: "local/audio-model",
        name: "Friendly audio model",
        isVision: false,
        isLora: true,
        isGguf: false,
        isAudio: true,
        audioType: "tts",
        hasAudioInput: false,
      },
    ],
  );
});

test("audio-input capability keeps the synthesized model off the output-only path", () => {
  const [model] = mergeQueuedModelCapabilities([], "local/audio-vlm", {
    isVision: false,
    isGguf: false,
    isAudio: true,
    audioType: "audio_vlm",
    hasAudioInput: true,
  });

  assert.equal(model.isAudio && !model.hasAudioInput, false);
});

test("synthesized audio-only capability reaches image validation", () => {
  const [model] = mergeQueuedModelCapabilities([], "local/audio-model", {
    isVision: false,
    isGguf: false,
    isAudio: true,
    audioType: "tts",
    hasAudioInput: false,
  });

  assert.equal(
    getImageInputUnavailableReason({
      activeModel: model,
      isExternalModel: false,
      loadedIsMultimodal: true,
      modelLoaded: true,
    }),
    "local/audio-model cannot accept images. Load a vision-capable model before attaching images.",
  );
});
