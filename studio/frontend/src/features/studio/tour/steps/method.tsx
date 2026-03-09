// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { ReadMore, type TourStep } from "@/features/tour";

export const studioMethodStep: TourStep = {
  id: "method",
  target: "studio-method",
  title: "Method: QLoRA vs LoRA vs Full",
  body: (
    <>
      LoRA: trains small adapter weights (fast, common default). QLoRA: LoRA on
      4-bit base weights (much lower VRAM). Full: updates all weights (highest
      cost, usually needs more data to be worth it).{" "}
      <ReadMore href="https://docs.unsloth.ai/basics/lora-hyperparameters-guide" />
    </>
  ),
};
