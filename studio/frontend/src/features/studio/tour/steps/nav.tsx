// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioNavStep: TourStep = {
  id: "nav",
  target: "navbar",
  title: "Quick orientation",
  body: (
    <>
      Studio: pick base model, dataset, hyperparams, then start training. After
      you start, you’ll see a Training view with live loss/metrics. Chat is for
      testing base vs LoRA adapters. Export packages checkpoints for deployment.{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-for-beginners" />
    </>
  ),
};
