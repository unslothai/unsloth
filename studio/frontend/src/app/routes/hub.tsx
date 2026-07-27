// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ModelsPage = lazyRouteComponent(
  () => import("@/features/hub/hub-page"),
  "ModelsPage",
);

export interface ModelsSearch {
  tab?: "discover" | "downloaded";
  model?: string;
  section?: "trending" | "latest" | "finetune";
  kind?: "models" | "datasets";
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/hub",
  beforeLoad: () => requireAuth(),
  component: ModelsPage,
  validateSearch: (search: Record<string, unknown>): ModelsSearch => {
    const next: ModelsSearch = {};
    const raw = search.tab;
    if (raw === "discover" || raw === "downloaded") next.tab = raw;
    const model = search.model;
    if (typeof model === "string" && model.length > 0) next.model = model;
    const section = search.section;
    if (
      section === "trending" ||
      section === "latest" ||
      section === "finetune"
    ) {
      next.section = section;
    }
    const kind = search.kind;
    if (kind === "models" || kind === "datasets") {
      next.kind = kind;
    }
    return next;
  },
});
