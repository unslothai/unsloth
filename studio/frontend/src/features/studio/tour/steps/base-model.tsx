// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioBaseModelStep: TourStep = {
  id: "base-model",
  target: "studio-model-picker",
  title: "Model",
  body: (
    <>
      Search Hugging Face, browse models already on this device, or paste{" "}
      <span className="font-mono">org/model</span>. Pick a base model close to
      your task (chat/instruct vs base). Smaller models iterate faster; scale up
      once prompts + data look good.{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use" />
    </>
  ),
};
