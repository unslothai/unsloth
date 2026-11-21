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
        URL and send the model name shown below it. Context and queue slots sit
        beside the model.
      </>
    ),
  },
  {
    id: "toolbar",
    target: "api-toolbar",
    title: "Model controls",
    body: (
      <>
        Load or switch the model the API serves, reload the last one, or eject
        it to free memory. Each model's row below opens its run settings and an
        API example. Keys and other API options are under API settings at the
        top.
      </>
    ),
  },
  {
    id: "log",
    target: "api-log",
    title: "Request log",
    body: (
      <>
        Pick a request to see its prompt, reply, token counts and any error.
        Search and the status filter narrow the list, and Live pauses the feed
        so a request does not scroll away.
      </>
    ),
  },
];
