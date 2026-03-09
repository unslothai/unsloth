// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { ReadMore, type TourStep } from "@/features/tour";

export const studioBaseModelStep: TourStep = {
  id: "base-model",
  target: "studio-base-model",
  title: "Base model from Hugging Face",
  body: (
    <>
      Paste <span className="font-mono">org/model</span> or search. Pick a base
      model close to your task (chat/instruct vs base). Smaller models iterate
      faster; scale up once prompts + data look good.{" "}
      <ReadMore href="https://docs.unsloth.ai/basics/fine-tuning-llms-guide" />
    </>
  ),
};
