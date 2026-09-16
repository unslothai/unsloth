// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioSaveStep: TourStep = {
  id: "save",
  target: "studio-save",
  title: "Save the config",
  body: (
    <>
      Save this setup as YAML, load it back later, or reset to defaults. Rerun
      the same baseline and it stays obvious whether a change actually helped.
    </>
  ),
};
