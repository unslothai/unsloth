// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { inferTrainingModelModalityFlags } from "../src/features/training/lib/model-modality-inference.ts";

test("ignores modality words in parent filesystem directories", () => {
  for (const identifier of [
    "/home/me/audio/models/Meta-Llama-3-8B",
    "/Users/me/vision/models/mistral-7b",
    "/data/speech/qwen2.5-7b",
    "C:/models/tts/llama-3.2-1b",
    String.raw`C:\audio\models\llama-3.2-1b`,
    "audio/Meta-Llama-3-8B",
  ]) {
    assert.deepEqual(
      inferTrainingModelModalityFlags({ identifiers: [identifier] }),
      { isAudio: false, isVision: false },
      identifier,
    );
  }
});

test("keeps modality hints from model names and metadata", () => {
  assert.deepEqual(
    inferTrainingModelModalityFlags({
      identifiers: ["/data/models/my-audio-model"],
    }),
    { isAudio: true, isVision: false },
  );
  assert.deepEqual(
    inferTrainingModelModalityFlags({
      identifiers: [String.raw`C:\models\my-vision-model`],
    }),
    { isAudio: false, isVision: true },
  );
  assert.deepEqual(
    inferTrainingModelModalityFlags({ identifiers: ["org/Qwen2.5-VL"] }),
    { isAudio: false, isVision: true },
  );
  assert.deepEqual(
    inferTrainingModelModalityFlags({
      tags: ["AUTOMATIC-SPEECH-RECOGNITION"],
      pipelineTag: "IMAGE-TO-TEXT",
    }),
    { isAudio: true, isVision: true },
  );
});
