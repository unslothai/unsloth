// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const EditRecipePage = lazy(() =>
  import("@/features/data-recipes").then((m) => ({
    default: m.EditRecipePage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data-recipes/$recipeId",
  beforeLoad: () => requireAuth(),
  component: DataRecipeEditorRoute,
});

function DataRecipeEditorRoute(): ReactElement {
  const { recipeId } = Route.useParams();
  return <EditRecipePage recipeId={recipeId} />;
}
