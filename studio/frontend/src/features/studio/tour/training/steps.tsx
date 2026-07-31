// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioTrainingTourSteps: TourStep[] = [
  {
    id: "nav",
    target: "navbar",
    title: "Training view",
    body: (
      <>
        This view updates live as training runs. Watch loss, speed, and ETA, and
        use Stop if you need to bail out or save.
      </>
    ),
  },
  {
    id: "progress",
    target: "studio-training-progress",
    title: "Progress + ETA",
    body: (
      <>
        Phase shows what we’re doing (loading model/dataset, configuring,
        training). ETA is rough early on; it stabilizes after a few steps.
      </>
    ),
  },
  {
    id: "train-loss",
    target: "studio-training-loss",
    title: "Training loss",
    body: (
      <>
        Training loss should generally trend down. Absolute values vary by
        dataset + tokenizer, so use it for direction more than “a magic number”.
        If loss goes very low (eg below ~0.2), that can be a sign you’re
        overfitting. If loss plateaus high, you likely need better data
        formatting, more data, or different hyperparams.
      </>
    ),
  },
  {
    id: "eval-loss",
    target: "studio-eval-loss",
    title: "Eval loss (validation)",
    body: (
      <>
        Eval loss is your sanity check. If training loss keeps dropping but eval
        loss goes up, you’re likely overfitting. To track it, set an eval dataset
        and `eval_steps` (setting `eval_steps=1` can be very slow).
      </>
    ),
  },
  {
    id: "stop",
    target: "studio-training-stop",
    title: "Stop / save",
    body: (
      <>
        Stop training any time. “Stop and Save” keeps the checkpoint/adapters so
        you can export or compare later.
      </>
    ),
  },
];
