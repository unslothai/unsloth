// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { TourStep } from "@/features/tour";

export const studioStartStep: TourStep = {
  id: "start",
  target: "studio-start",
  title: "Start training",
  body: (
    <>
      Kick off training. If it errors immediately, check HF token / local paths
      / dataset access first. Start with a small run to sanity-check loss + sample
      outputs before burning hours.
    </>
  ),
};
