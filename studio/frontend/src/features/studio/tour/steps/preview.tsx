// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

/** The Start button sits inside this card, so one step covers the check and the launch. */
export const studioPreviewStep: TourStep = {
  id: "preview",
  target: "studio-run-preview",
  title: "Check, then start",
  body: (
    <>
      Step count, batch size, context and detected hardware, plus anything that
      downloads on start. Once the Ready pill is on, Start opens the run in
      Current Run and it keeps going if you switch pages. Do a short run first.
    </>
  ),
};
