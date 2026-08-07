// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// RootLayout renders AudioPage persistently (so an in-flight generation is not cancelled when leaving the tab); this route only owns the URL + auth gate.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/audio",
  staticData: { title: "Audio" },
  // An audio pick made from the chat picker arrives here as ?model= (+ ?quant= and task), which the page loads and then clears.
  validateSearch: (
    search: Record<string, unknown>,
  ): { model?: string; quant?: string; task?: string } => ({
    ...(typeof search.model === "string" ? { model: search.model } : {}),
    ...(typeof search.quant === "string" ? { quant: search.quant } : {}),
    ...(search.task === "automatic-speech-recognition" ||
    search.task === "text-to-speech"
      ? { task: search.task }
      : {}),
  }),
  beforeLoad: () => requireAuth(),
  component: () => null,
});
