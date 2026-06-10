// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ModelsPage = lazy(() =>
  import("@/features/hub/hub-page").then((m) => ({
    default: m.ModelsPage,
  })),
);

export interface ModelsSearch {
  tab?: "discover" | "downloaded";
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/hub",
  beforeLoad: () => requireAuth(),
  component: ModelsPage,
  validateSearch: (search: Record<string, unknown>): ModelsSearch => {
    const raw = search.tab;
    if (raw === "discover" || raw === "downloaded") return { tab: raw };
    return {};
  },
});
