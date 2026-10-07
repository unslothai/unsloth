// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/video",
  staticData: { title: "Video" },
  // ?model= (+ ?quant= exact filename, ?ggufQuant= label), loaded then cleared by the page.
  validateSearch: (
    search: Record<string, unknown>,
  ): { model?: string; quant?: string; ggufQuant?: string; item?: string } => ({
    ...(typeof search.model === "string" ? { model: search.model } : {}),
    ...(typeof search.quant === "string" ? { quant: search.quant } : {}),
    ...(typeof search.ggufQuant === "string"
      ? { ggufQuant: search.ggufQuant }
      : {}),
    ...(typeof search.item === "string" ? { item: search.item } : {}),
  }),
  beforeLoad: () => requireAuth(),
  component: () => null,
});
