// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioTrainingTourSteps: TourStep[] = [
  {
    id: "progress",
    target: "studio-training-progress",
    title: "Progress and ETA",
    body: (
      <>
        Phase says what is happening right now, from loading the model and
        dataset through to training. The ETA is rough at first and settles after
        a few steps. Leaving this page does not stop the run.
      </>
    ),
  },
  {
    id: "train-loss",
    target: "studio-training-loss",
    title: "Training loss",
    body: (
      <>
        Watch the direction, not the number, since the scale shifts with your
        dataset and tokenizer. Flat and high usually means data formatting or
        hyperparameters. Very low, say under 0.2, usually means overfitting.
      </>
    ),
  },
  {
    id: "eval-loss",
    target: "studio-eval-loss",
    title: "Eval loss",
    body: (
      <>
        Your sanity check. Training loss falling while eval loss climbs is
        overfitting. It needs an eval dataset and an eval_steps value, set in
        Advanced parameters. Small eval_steps values slow the run down a lot.
      </>
    ),
  },
  {
    id: "stop",
    target: "studio-training-stop",
    title: "Stop and save",
    body: (
      <>
        Stop any time. Stop and Save keeps the checkpoint and adapters, so you
        can test them in Chat or package them on the Export page.
      </>
    ),
  },
];
