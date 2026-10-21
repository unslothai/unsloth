// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioParamsStep: TourStep = {
  id: "params",
  target: "studio-params",
  title: "Parameters",
  body: (
    <>
      Simple covers epochs, learning rate, batch size and context length.
      Advanced opens LoRA rank, schedulers, eval and checkpointing. Defaults
      come from the model you picked, so change one knob at a time. 1 to 3
      epochs is a good start.{" "}
      <ReadMore href="https://unsloth.ai/docs/basics/lora-hyperparameters-guide" />
    </>
  ),
};
