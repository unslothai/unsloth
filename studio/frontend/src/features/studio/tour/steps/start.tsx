// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioStartStep: TourStep = {
  id: "start",
  target: "studio-start",
  title: "Start training",
  body: (
    <>
      The run opens in Current Run and keeps going if you switch pages. Do a
      short run first to check the loss curve and a few sample outputs. If it
      fails right away, look at your HF token, local paths and dataset access.
    </>
  ),
};
