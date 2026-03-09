// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const DataRecipesPage = lazy(() =>
  import("@/features/data-recipes").then((m) => ({
    default: m.DataRecipesPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data-recipes",
  beforeLoad: () => requireAuth(),
  component: DataRecipesPage,
});
