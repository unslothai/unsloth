// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioNavStep: TourStep = {
  id: "nav",
  target: "studio-subnav",
  title: "Three tabs",
  body: (
    <>
      Configure sets up a run. Current Run shows live loss and metrics once it
      starts. History keeps every past run, so you can reopen, resume or export
      one. Image and diffusion training lives on the Images page.{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-for-beginners" />
    </>
  ),
};
