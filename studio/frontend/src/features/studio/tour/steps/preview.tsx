// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioPreviewStep: TourStep = {
  id: "preview",
  target: "studio-run-preview",
  title: "Run preview",
  body: (
    <>
      Read this before you start. It shows step count, batch size, context and
      the hardware we detected, and it flags anything that downloads on start.
      The Ready pill turns on once the run has everything it needs.
    </>
  ),
};
