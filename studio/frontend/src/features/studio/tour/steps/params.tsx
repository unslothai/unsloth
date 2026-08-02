// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioParamsStep: TourStep = {
  id: "params",
  target: "studio-params",
  title: "Dial hyperparams",
  body: (
    <>
      Start boring, then iterate. We usually recommend starting with 1-3 epochs
      (higher can overfit fast). If you’re unsure, change 1 knob at a time, and
      watch train vs eval loss.{" "}
      <ReadMore href="https://unsloth.ai/docs/basics/lora-hyperparameters-guide" />
    </>
  ),
};
