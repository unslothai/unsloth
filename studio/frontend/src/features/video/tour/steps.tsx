// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const videoTourSteps: TourStep[] = [
  {
    id: "model",
    target: "video-model",
    title: "Pick a model",
    body: (
      <>
        Video models run locally, so check the size before you download. What
        the model supports, such as keyframes, reference images or audio,
        decides which controls appear below.
      </>
    ),
  },
  {
    id: "prompt",
    target: "video-prompt",
    title: "Prompt",
    body: (
      <>
        Describe the shot and the motion, not just the subject. Add a start or
        end frame below to drive it from an image you already have.
      </>
    ),
  },
  {
    id: "settings",
    target: "video-settings",
    title: "Settings",
    body: (
      <>
        Resolution, duration in frames, steps, guidance and seed. Duration and
        resolution drive VRAM and time more than anything else, so start small.
      </>
    ),
  },
  {
    id: "generate",
    target: "video-generate",
    title: "Generate",
    body: (
      <>
        Starts the run and turns into Cancel while it works. The model loads on
        first use, so the first clip takes noticeably longer.
      </>
    ),
  },
  {
    id: "preview",
    target: "video-preview",
    title: "Results",
    body: (
      <>
        The selected clip plays here and every clip lines up in the strip below.
        Download it, or reuse its settings for the next run.
      </>
    ),
  },
];
