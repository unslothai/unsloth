// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioBaseModelStep: TourStep = {
  id: "base-model",
  target: "studio-model-picker",
  title: "Base model",
  body: (
    <>
      Search Hugging Face, reuse a model already on this device, or paste{" "}
      <span className="font-mono">org/model</span>. Picking one auto-fills
      sensible hyperparameters. Smaller models iterate faster, so scale up once
      the data looks right. GGUF files cannot be trained.{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use" />
    </>
  ),
};
