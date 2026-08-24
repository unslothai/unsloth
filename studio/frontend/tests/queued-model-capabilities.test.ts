import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

// image-input-support.ts imports ./mmproj-fallback without an extension, which is
// what 2314 of the 2367 relative imports under src/ do -- vite and tsconfig's
// "bundler" mode resolve it, the bare node loader does not. A static import here
// resolves before any registration can run, so the module has to come in
// dynamically, after the resolver is registered.
registerBundlerResolver();

const { getImageInputUnavailableReason } = await import(
  "../src/features/chat/utils/image-input-support.ts"
);
const { mergeQueuedModelCapabilities } = await import(
  "../src/features/chat/utils/queued-model-capabilities.ts"
);

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
