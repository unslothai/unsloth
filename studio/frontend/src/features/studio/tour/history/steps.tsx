// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";
import { studioNavStep } from "../steps/nav";

export const studioHistoryTourSteps: TourStep[] = [
  studioNavStep,
  {
    id: "history",
    target: "studio-history",
    title: "Past runs",
    body: (
      <>
        Every finished or stopped run, with its config and loss curves. Open one
        to see how it went, resume it from its last checkpoint, or send its
        checkpoints to Export.
      </>
    ),
  },
];
