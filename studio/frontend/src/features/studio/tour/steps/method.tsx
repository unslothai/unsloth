// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioMethodStep: TourStep = {
  id: "method",
  target: "studio-method",
  title: "Training method",
  body: (
    <>
      QLoRA trains adapters on a 4-bit base and needs the least VRAM. LoRA does
      the same in 16-bit. Full fine-tune updates every weight. Continued
      pretraining is for teaching a new domain or language.{" "}
      <ReadMore href="https://unsloth.ai/docs/basics/lora-hyperparameters-guide" />
    </>
  ),
};
