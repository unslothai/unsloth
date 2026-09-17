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
        the model supports decides which controls appear below.
      </>
    ),
  },
  {
    id: "settings",
    target: "video-settings",
    title: "Settings",
    body: (
      <>
        Prompt, resolution, duration in frames, steps, guidance and seed.
        Duration and resolution drive VRAM and time most, so start small.
      </>
    ),
  },
  {
    id: "preview",
    target: "video-preview",
    title: "Results",
    body: (
      <>
        The selected clip plays here and every clip lines up below. Download it,
        or reuse its settings for the next run.
      </>
    ),
  },
];
