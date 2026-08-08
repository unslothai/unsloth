// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// RootLayout renders ImagesPage persistently (so an in-flight batch is not cancelled when leaving the tab); this route only owns the URL + auth gate.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/images",
  staticData: { title: "Images" },
  // A diffusion pick made from the chat picker arrives here as ?model= (+ ?quant= for an exact filename, or ?ggufQuant= for a
  // label the page still has to resolve), which the page loads and then clears.
  validateSearch: (
    search: Record<string, unknown>,
  ): { model?: string; quant?: string; ggufQuant?: string } => ({
    ...(typeof search.model === "string" ? { model: search.model } : {}),
    ...(typeof search.quant === "string" ? { quant: search.quant } : {}),
    ...(typeof search.ggufQuant === "string"
      ? { ggufQuant: search.ggufQuant }
      : {}),
  }),
  beforeLoad: () => requireAuth(),
  component: () => null,
});
