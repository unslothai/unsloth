// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { isDiffusionTask } from "../src/features/model-picker/components/model-selector/model-capabilities.ts";

test("Chat excludes both downloaded FLUX formats while keeping text and VLMs", () => {
  const cached = [
    { id: "unsloth/flux-3-image-klein-pup-GGUF", task: "text-to-image" },
    { id: "unsloth/flux-3-image-klein-FP8", task: "text-to-image" },
    { id: "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF", task: "text-generation" },
    { id: "unsloth/Qwen3.5-2B", task: "image-text-to-text" },
  ];
  assert.deepEqual(cached.filter((row) => !isDiffusionTask(row.task)).map((row) => row.id), [
    "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF",
    "unsloth/Qwen3.5-2B",
  ]);
});

test("image, editing and video generation tasks stay out of Chat", () => {
  for (const task of [
    "text-to-image", "image-to-image", "image-text-to-image", "inpainting",
    "text-to-video", "image-to-video", "image-text-to-video", "image-diffusion-unsupported",
  ]) {
    assert.equal(isDiffusionTask(task), true, task);
  }
});

test("unknown metadata and existing audio routing retain their behavior", () => {
  for (const task of [undefined, null, "text-generation", "image-text-to-text", "text-to-speech"]) {
    assert.equal(isDiffusionTask(task), false, String(task));
  }
});
