// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const apiMonitorTourSteps: TourStep[] = [
  {
    id: "endpoint",
    target: "api-endpoint",
    title: "Your endpoint",
    body: (
      <>
        Unsloth serves an OpenAI-compatible API. Point any client at this base
        URL, keep the model name the same, and it works. Status and queue slots
        sit beside it.
      </>
    ),
  },
  {
    id: "toolbar",
    target: "api-toolbar",
    title: "Controls",
    body: (
      <>
        Pause the live feed to read a request without it scrolling away. Unload
        frees the model's VRAM. API settings is where you create keys.
      </>
    ),
  },
  {
    id: "log",
    target: "api-log",
    title: "Request log",
    body: (
      <>
        Pick a request on the left to see its prompt, reply, token counts and
        any error on the right. Search and the status filter above narrow it
        down.
      </>
    ),
  },
];
