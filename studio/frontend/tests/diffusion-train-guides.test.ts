// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DIFFUSION_TRAIN_GUIDE_URLS,
  diffusionTrainGuideUrl,
} from "../src/features/images/train/diffusion-train-guides.ts";

const families = [
  { name: "sdxl", is_video: false },
  { name: "flux", is_video: false },
  { name: "wan22-5b", is_video: true },
  { name: "minimax-h3" },
  { name: "ltx2", is_video: true },
];

test("video families link the video LoRA guide", () => {
  assert.equal(diffusionTrainGuideUrl("wan22-5b", families), DIFFUSION_TRAIN_GUIDE_URLS.video);
  assert.equal(diffusionTrainGuideUrl("ltx2", families), DIFFUSION_TRAIN_GUIDE_URLS.video);
  assert.equal(
    DIFFUSION_TRAIN_GUIDE_URLS.video,
    "https://unsloth.ai/docs/basics/fine-tune-video-models-with-lora",
  );
});

test("image families link the diffusion LoRA guide", () => {
  assert.equal(diffusionTrainGuideUrl("sdxl", families), DIFFUSION_TRAIN_GUIDE_URLS.image);
  assert.equal(diffusionTrainGuideUrl("flux", families), DIFFUSION_TRAIN_GUIDE_URLS.image);
  assert.equal(diffusionTrainGuideUrl("minimax-h3", families), DIFFUSION_TRAIN_GUIDE_URLS.image);
  assert.equal(
    DIFFUSION_TRAIN_GUIDE_URLS.image,
    "https://unsloth.ai/docs/basics/fine-tune-diffusion-models-with-lora",
  );
});

test("unknown, empty or unloaded families fall back to the diffusion guide", () => {
  assert.equal(diffusionTrainGuideUrl("not-a-family", families), DIFFUSION_TRAIN_GUIDE_URLS.image);
  assert.equal(diffusionTrainGuideUrl("", families), DIFFUSION_TRAIN_GUIDE_URLS.image);
  assert.equal(diffusionTrainGuideUrl(null, families), DIFFUSION_TRAIN_GUIDE_URLS.image);
  assert.equal(diffusionTrainGuideUrl("wan22-5b", undefined), DIFFUSION_TRAIN_GUIDE_URLS.image);
  assert.equal(diffusionTrainGuideUrl("wan22-5b", []), DIFFUSION_TRAIN_GUIDE_URLS.image);
});
